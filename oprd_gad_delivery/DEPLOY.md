# DEPLOY — OPRD + GAD on a new machine

End-to-end deployment on a **fresh machine with the same CUDA driver and same GPU model** (8× H20).
Covers: environment → models → data → smoke test → experiments.

> Layout note: this delivery is self-contained. `bootstrap` clones the OPRD fork into
> `<delivery>/.oprd` and applies our merge there. `/dockerdata` is **not** shared across machines,
> so **models and datasets must be transferred/downloaded per machine** (they are not in the delivery).

---

## Prerequisites
- `conda`, `git`, and an NVIDIA driver (same as the source box; `nvidia-smi` "CUDA 12.4" is the driver
  max — torch 2.8 ships its own runtime and is compatible, no system CUDA toolkit needed).
- Enough disk for models (Qwen3-32B ≈ 64 GB, Qwen3-4B ≈ 8 GB), data, and training outputs.

## Step 1 — Get the delivery
```bash
git clone <your LMOps repo> /path/to/LMOps        # or: git pull
cd /path/to/LMOps/oprd_gad_delivery
```

## Step 2 — Bootstrap (code + conda env), one command
```bash
CONDA_ENV=oprd_gad bash bootstrap_oprd_gad.sh
```
This runs `clone patch env check`:
1. **clone** OPRD @ `BASE_COMMIT` (93816fd) into `<delivery>/.oprd`.
2. **patch** — overlay our merge (`modified_files_full/` + `scripts/*`, incl. `.py` helpers) onto the checkout.
3. **env** — create conda env, install the verl stack (vllm 0.11 / torch 2.8 / flash-attn 2.8.1 / ray),
   then `pip install scipy==1.15.3 matplotlib` (the stack pulls scipy 1.18 which crashes `import
   transformers` under numpy 1.26; matplotlib is needed when `is_plot=True`).
4. **check** — verify torch+CUDA, import vllm/flash_attn.

Env overrides: `CONDA_ENV` (default `verl`), `TARGET_DIR` (default `<delivery>/.oprd`),
`PATCH_MODE` (`overlay`|`apply`), `OPRD_REPO_URL`, `BASE_COMMIT`.

Then work inside the checkout:
```bash
cd /path/to/LMOps/oprd_gad_delivery/.oprd
conda activate oprd_gad
```

## Step 3 — Models (transfer; not in delivery)
Get the models onto this box (pick one):
```bash
# A) rsync from the source machine (fastest on the same cluster)
rsync -a <src>:/dockerdata/junewluo/models/Qwen3-4B  /dockerdata/junewluo/models/
rsync -a <src>:/dockerdata/junewluo/Qwen3-32B         /dockerdata/junewluo/
# (for the smoke test also need a tiny model, e.g. Qwen2.5-0.5B-Instruct, under MODEL_DIR)

# B) or download the corresponding models from HuggingFace into the same paths
```
Default paths: student `MODEL_DIR=/dockerdata/junewluo/models` (`Qwen3-4B`), teacher
`/dockerdata/junewluo/Qwen3-32B`. Otherwise override via env / Hydra at run time.

## Step 4 — Data (download + dedup)
```bash
export DATA_ROOT=/dockerdata/junewluo/datasets     # this box's data dir (launch scripts read this)
bash download_data.sh                              # DAPO-Math-17k (train) + AIME24 (eval) -> $DATA_ROOT
python3 dedup_train_parquet.py \
  --in  $DATA_ROOT/dapo-math-17k.parquet \
  --out $DATA_ROOT/dapo-math-17k-dedup.parquet     # 1,791,700 rows -> 17,398 unique prompts

# For the OPD / OPRD baselines (same 2000 prompts as GAD, no teacher_response column):
python3 -c "import pyarrow.parquet as pq; pq.write_table(pq.read_table('$DATA_ROOT/dapo-math-17k-dedup.parquet').slice(0,2000), '$DATA_ROOT/dapo-math-17k-dedup-2000.parquet')"
```
> GAD's `teacher_response` column is built **automatically** on the first GAD run (step 0 inside
> `gad_oprd_distillation.sh`, using the 32B teacher on the dedup set). To pre-build (optionally random-
> sampled to avoid DAPO's answer-magnitude ordering bias):
> `python3 build_teacher_response_parquet.py --in $DATA_ROOT/dapo-math-17k-dedup.parquet --out $DATA_ROOT/dapo-math-17k-dedup-gad.parquet --teacher /dockerdata/junewluo/Qwen3-32B --n 2000 --tp 4 --sample --seed 42`

## Step 5 — Smoke test (confirm end-to-end first)
```bash
export DATA_ROOT=/dockerdata/junewluo/datasets
export TEST_FILE='["'$DATA_ROOT'/test_data/AIME24/test.parquet"]'
bash smoke_debug.sh          # 1 GPU, ~0.5B models, resp 512, 3 steps (GAD; step0 builds teacher_response with the 0.5B model)
```
Proceed only after it reaches step 1 with metrics and no errors.

## Step 6 — Real experiments (3-way, aligned eval)
Common environment:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # mitigate GPU-mem fragmentation
export DATA_ROOT=/dockerdata/junewluo/datasets
export MODEL_DTYPE=bfloat16
export MAX_RESP_LENGTH=8192 MAX_VAL_RESP_LENGTH=8192
export N_RESPONSES=2 VAL_N=2 MINI_BATCH_SIZE=32 TEST_FREQ=20 SAVE_FREQ=62
export PROJECT_PATH=/dockerdata/junewluo/runs/oprd_gad      # keep outputs OUT of .oprd
export TEST_FILE='["'$DATA_ROOT'/test_data/AIME24/test.parquet"]'
COMMON="actor_rollout_ref.model.path=/dockerdata/junewluo/models/Qwen3-4B \
        reward_model.model.path=/dockerdata/junewluo/Qwen3-32B \
        data.dataloader_num_workers=1 trainer.val_before_train=True trainer.total_epochs=5"
```
Run one at a time (don't share GPUs across the three):
```bash
# (1) OPRD + GAD, with GAN/GAIL stabilization tricks
GAD_REWARD_SHAPING=gail GAD_D_GATE=True KL_LOSS_COEF=0.03 CRITIC_LR=5e-7 GAD_COEF=0.1 \
REP_DISTILLATION_LAYERS=last REP_DISTILLATION_LAST_K=256 TEACHER_PARAM_OFFLOAD=False \
nohup bash gad_oprd_distillation.sh $COMMON > logs/nohup_gad.log 2>&1 &

# (2) OPRD-only baseline
REP_DISTILLATION_LAYERS=last REP_DISTILLATION_LAST_K=256 TEACHER_PARAM_OFFLOAD=False \
nohup bash baseline_oprd_only.sh $COMMON \
  data.train_files=$DATA_ROOT/dapo-math-17k-dedup-2000.parquet > logs/nohup_oprd.log 2>&1 &

# (3) OPD baseline
LOG_PROB_TOP_K=16 \
nohup bash baseline_opd.sh $COMMON \
  data.train_files=$DATA_ROOT/dapo-math-17k-dedup-2000.parquet > logs/nohup_opd.log 2>&1 &
```
`nohup ... &` survives SSH disconnect (`echo $!` prints the PID). GAD uses its own train set
(`GAD_TRAIN_DATASET`, default `$DATA_ROOT/dapo-math-17k-dedup-gad.parquet`, auto-built at step 0).

## Monitor
```bash
python3 analyze_train_log.py $(ls -t logs/run_*.log | head -1)   # eval curve + training trend + gate activity
tail -f $(ls -t logs/run_*.log | head -1)                        # raw live log
```

---

## Gotchas
1. **Shared GPUs** — `nvidia-smi` before launching; other tenants (other containers) may occupy the cards.
2. **Models & data must exist first** (Steps 3–4) or Steps 5–6 fail on missing files / OOM.
3. **`TEST_FILE` default is `AMC23` (not present)** — always point it at AIME24 (as above).
4. **GPU memory (32B teacher + resp 8192):** `MINI_BATCH_SIZE=32` is stable; `64` OOM'd at ~step 19
   (fragmentation at the ~93/95 GB edge). If tight, lower `MINI_BATCH_SIZE`, then `MAX_RESP_LENGTH`,
   then `REP_DISTILLATION_LAST_K`; keep `TEACHER_PARAM_OFFLOAD=False` (teacher stays sharded on GPU,
   not offloaded to CPU RAM).
5. **`.oprd` is a per-machine runtime checkout** — don't ship it in the delivery bundle (bootstrap
   recreates it). Set `PROJECT_PATH` outside `.oprd` so training outputs don't bloat the delivery dir.
6. **`DATA_ROOT` / `MODEL_DIR`** default to `/dockerdata/junewluo/...`; set them if this box uses other paths.

See `README.md` (§Updates) for what changed vs the original delivery and the method/architecture details.
