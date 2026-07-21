# OPRD + GAD — Combined White-box Cross-arch Distillation

> 中文版: [README.zh.md](README.zh.md)

Merges **GAD** (Generative Adversarial Distillation — a discriminator reward optimized by GRPO)
into the **OPRD** (On-Policy Representation Distillation — teacher hidden-state MSE) codebase.

**Method:** representation distillation is the *primary* signal; the adversarial discriminator
reward is *auxiliary*.

```
L_actor =  rep_distillation_coef * MSE(h_student, sg(h_teacher))     # OPRD — primary, deterministic
        +  gad_coef             * PG(D(y_student))  via GRPO         # GAD  — auxiliary, adversarial
        +  kl_loss_coef         * KL(student || ref)                 # anchor
```

- **Setting:** white-box teacher (e.g. Qwen3-4B) → small student (e.g. Qwen3-1.7B), math benchmarks.
- **Baselines (native, unchanged):** OPRD-only, OPD.
- **Positioning:** this is a *white-box* method (needs teacher hidden states); it is **not** compared
  against black-box GAD. GAD contributes only its adversarial-reward mechanism.

## Updates (2026-07): pipeline hardening + GAN/GAIL tricks + deployment

Changes made after getting the pipeline running end-to-end on 8× H20 (Qwen3-32B teacher → Qwen3-4B student). All new behavior is **opt-in / default-preserving** unless noted; OPD / OPRD baselines stay isolated.

**Environment fix (folded into `bootstrap`).** The vLLM stack pulls **scipy 1.18** (needs numpy≥2), which crashes `import transformers` under the pinned numpy 1.26 via `np.long`. The env stage now runs `pip install scipy==1.15.3 matplotlib` (matplotlib is needed when `is_plot=True`).

**GAN/GAIL stabilization tricks (opt-in, default OFF)** — added to fight discriminator saturation / divergence (an un-tricked 32B→4B run had `d_acc→1.0`, KL blow-up, student AIME24 acc collapsing 0.16→0):
| env | default | effect |
|---|---|---|
| `GAD_REWARD_SHAPING` | `raw` | `gail` = bounded `logσ(D)` reward (raw `D` is unbounded, explodes when D saturates) |
| `GAD_D_GATE` | `False` | adaptive discriminator gating: skip the D update when last `d_acc > GAD_D_ACC_HI` |
| `GAD_D_ACC_HI` | `0.6` | gate threshold |
| `GAD_D_MAX_SKIP` | `5` | **failsafe**: never skip D more than N steps in a row (so D can't be starved) |

New metrics: `gad/d_update_skipped`, `gad/d_skip_count`. All gated behind `use_gad_discriminator`. Code: reward shaping + gating in `ray_trainer.py`; fields declared in `workers/config/actor.py`.

**`teacher_response` no longer crashes on length.** `rl_dataset.py` now tokenizes `teacher_response` with `truncation="right"` (was the global `truncation='error'` → crashed when a teacher solution exceeded `max_response_length`). Over-long teacher references are truncated, not fatal. *(Supersedes the old "over-long teacher_response will crash" gotcha below.)*

**`gad_oprd_distillation.sh` self-contains step 0.** It auto-builds the `teacher_response` parquet if missing (`GAD_TRAIN_DATASET`), injects student/teacher/discriminator model paths via Hydra, and defaults `MODEL_DTYPE=bfloat16`. Env knobs: `TEACHER_MODEL_PATH` (default Qwen3-32B), `STUDENT_MODEL_PATH` (default Qwen3-4B), `GAD_BASE_DATASET`, `TEACHER_GEN_N`, `TEACHER_GEN_TP`, `FORCE_REBUILD_TEACHER`.

**`build_teacher_response_parquet.py`** gained `--tp N` (tensor-parallel for large teachers, e.g. 32B) and `--sample [--seed]` (random-sample N rows — DAPO is **ordered by answer magnitude**, so the first-N is biased).

**More env-configurable knobs** (`on_policy_distillation.sh`): `KL_LOSS_COEF`/`KL_LOSS_TYPE` (default 0.005 / low_var_kl); `VAL_N` (val n, default **4**, was 16); `MAX_VAL_RESP_LENGTH` (default **8192**, was 15360); `TEST_FREQ` (default **50**, was 2). OPRD-only: `REP_DISTILLATION_LAST_K`/`REP_DISTILLATION_LAYERS` now overridable.

**Data paths are now absolute (`DATA_ROOT`).** `TRAIN_DATASET`, `TEST_DATA_DIR`, `GAD_BASE_DATASET` derive from `DATA_ROOT` (default `/dockerdata/junewluo/datasets`) instead of the CWD-relative `../datasets` — so runs work regardless of where the repo is checked out. Override with `export DATA_ROOT=/that/box/datasets`.

**Reward-function path fixed** in `on_policy_distillation.sh`: `verl/utils/...` → `verl/verl/utils/...` (nested layout). No longer needs a command-line override.

**Deployment = self-contained checkout.** `bootstrap` `TARGET_DIR` default is now `<delivery>/.oprd` (was `$HOME/OPRD_gad`); the overlay step now `cp scripts/*` (was `*.sh`) so the `.py` helpers land in the repo root too. Notes: set `PROJECT_PATH` to keep the 100s-of-GB training outputs **out** of `.oprd`; don't ship a pre-cloned `.oprd` (bootstrap re-creates it per machine); gitignore `.oprd/` if the delivery is tracked.

**New helper scripts** (`scripts/`): `dedup_train_parquet.py` (DAPO ships ~100× duplicated — 1,791,700 rows / 17,398 unique prompts; dedup before training) and `analyze_train_log.py` (`python3 analyze_train_log.py logs/run_*.log` → eval curve + training-metric trend + gate activity).

**Memory profile (32B teacher → 4B student, 8× H20).** Full rep-MSE over all layers + `last_k=2000` OOMs; use `REP_DISTILLATION_LAYERS=last`, `REP_DISTILLATION_LAST_K=256`, `TEACHER_PARAM_OFFLOAD=False`, `MODEL_DTYPE=bfloat16`, `data.dataloader_num_workers=1`. `MINI_BATCH_SIZE=64` OOM'd at step 19 (fragmentation at the ~93/95 GB edge); `32` is safer, and `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` mitigates fragmentation. Note these are **shared** GPUs — check `nvidia-smi` for other tenants before launching.

---

## Contents

| Path | What |
|---|---|
| `DEPLOY.md` | **New-machine deployment runbook** — env → models → data → smoke → experiments. |
| `bootstrap_oprd_gad.sh` | **One-command** setup+run: clone OPRD@base → apply merge → build conda env → self-check → run. |
| `BASE_COMMIT` | The OPRD base commit the merge is pinned to (read by the bootstrap script). |
| `oprd_gad.patch` | All source changes, as a `git apply`-able patch against the OPRD fork (base commit `93816fd`). |
| `scripts/gad_oprd_distillation.sh` | **Combined OPRD+GAD** launcher. |
| `scripts/baseline_oprd_only.sh` | OPRD-only baseline. |
| `scripts/baseline_opd.sh` | OPD (and OPD+OPRD) baseline. |
| `scripts/smoke_debug.sh` | Tiny single-GPU smoke + post-mortem-pdb debug run (~3 steps). |
| `scripts/build_teacher_response_parquet.py` | Helper to add the `teacher_response` column. |
| `scripts/download_data.sh` | Download DAPO-Math-17k (train) + AIME24 (eval), verl-format. |
| `tests/test_gad_components.py` | CPU unit tests (BT loss + last-token masking); runs without GPU. |
| `modified_files_full/` | The 10 modified files as **complete files** (read-only reference; the patch is what you apply). |

## Setup (on the GPU machine)

The OPRD fork uses a **newer stack than GAD** (Python 3.12 / vLLM 0.11 / torch 2.8 /
flash-attn 2.8.1) — do **not** reuse GAD's docker image.

### One-command bootstrap (recommended)

From this delivery folder on the GPU machine:

```bash
bash bootstrap_oprd_gad.sh          # clone OPRD@BASE_COMMIT + apply merge + build conda env + self-check
# then, after preparing data with a teacher_response column:
RUN_SCRIPT=smoke_debug.sh bash bootstrap_oprd_gad.sh run
```

It clones OPRD pinned to `BASE_COMMIT`, **overlays** the merge (file-copy — CRLF/patch-context proof),
copies the launchers, builds the `verl` conda env, and runs the checks. Idempotent and stage-selectable:
`bash bootstrap_oprd_gad.sh clone patch | env | check | run | all`. Override via env:
`TARGET_DIR` (default `<delivery>/.oprd`), `CONDA_ENV` (`verl`), `PATCH_MODE` (`overlay`|`apply`),
`RUN_SCRIPT`, `RUN_ARGS`, `OPRD_REPO_URL`.

The manual steps below do the same thing by hand.

```bash
# 1. Get the source = clone the OPRD fork + apply our patch
git clone https://github.com/ShenzhiYang2000/OPRD.git
cd OPRD
git checkout 93816fd                                     # verl 0.7.0.dev base the patch targets
# run git apply from the OPRD repo ROOT (patch paths are verl/verl/... and on_policy_distillation.sh)
git apply /path/to/oprd_gad_delivery/oprd_gad.patch
cp /path/to/oprd_gad_delivery/scripts/*.sh .             # launchers, next to on_policy_distillation.sh

# 2. Environment (OPRD's official steps)
conda create -n verl python==3.12 -y
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh # vllm0.11 / torch2.8 / flash-attn2.8.1 / flashinfer / ray ...
pip install math-verify
pip install -e . --no-deps                               # make `verl` importable for `python -m verl.trainer.main_ppo`
cd ..

# 3. Model & data paths
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/datasets
```

Notes:
- `USE_MEGATRON=0` — we use FSDP, not Megatron (skips hard-to-build deps).
- Real experiments need **multiple GPUs** (scripts default to 8); a single GPU is only enough for the smoke run.

### CUDA / driver compatibility (target box: driver 535.161.08, "CUDA 12.4", 8× H20 96 GB)

- The `CUDA Version: 12.4` shown by `nvidia-smi` is the **maximum CUDA runtime the driver supports**, not a
  hard cap. PyTorch/vLLM ship their **own** CUDA runtime; thanks to **CUDA 12.x minor-version compatibility**,
  the stack this script installs (torch 2.8 with a cu126/cu128 build, vLLM 0.11) runs fine on a 12.4 (r535)
  driver. H20 (Hopper, `sm_90`) is fully supported.
- So install the script **as-is** — no special CUDA-12.4 build is needed, and do **not** install a system
  CUDA 12.4 toolkit (torch ignores it; `USE_MEGATRON=0` avoids the from-source builds that would need one).
- Verify right after install:
  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  python -c "import flash_attn, vllm; print('flash_attn', flash_attn.__version__, '| vllm', vllm.__version__)"
  ```
- Only if you hit `CUDA driver version is insufficient for CUDA runtime version` (unlikely on r535): reinstall
  torch's lower CUDA-minor build of the **same** version, e.g.
  `pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126`, and match the flash-attn /
  flashinfer / vLLM builds to it. Prefer this only if the default actually fails.
- 8× H20 (96 GB) is ample: run real experiments on all 8 (`N_GPUS_PER_NODE=8`, the script default); the smoke run uses 1.

## Data

**Train = DAPO-Math-17k; eval = AIME24** — both already in **verl format** on HuggingFace (no
preprocessing). Note AIME24 is a *validation* benchmark, not training data.

**Schema (verl format).** Each row:
```python
{
  "data_source": "math_dapo",
  "prompt": [{"role": "user", "content": "...  output the final answer within \\boxed{}."}],  # chat list; prompt_key="prompt"
  "ability": "math",
  "reward_model": {"style": "rule", "ground_truth": "42"},   # used by validation & the OPD/OPRD baselines
  "extra_info": {"index": 0, "split": "train"},
  "teacher_response": "We start by ... therefore \\boxed{42}."   # TRAIN-only, added for GAD (discriminator's "real" example)
}
```
Validation parquets (AIME24) do **not** need `teacher_response` (all reads are guarded). It is tokenized
with the **student** tokenizer and right-padded with eos to `data.max_response_length`.

**Download** (run from the OPRD repo root so `../datasets` resolves; idempotent):
```bash
bash scripts/download_data.sh
#  -> ../datasets/dapo-math-17k.parquet   and   ../datasets/test_data/AIME24/test.parquet
export TEST_FILE='["../datasets/test_data/AIME24/test.parquet"]'   # tell the launcher to eval on AIME24
```
(equivalently, `verl/recipe/dapo/prepare_dapo_data.sh` wgets the same two HF parquets.)

**Add the `teacher_response` column (GAD only):**
```bash
python3 scripts/build_teacher_response_parquet.py \
  --in ../datasets/dapo-math-17k.parquet --out ../datasets/dapo-math-17k-gad.parquet \
  --teacher $REWARD_MODEL_PATH
TRAIN_DATASET=../datasets/dapo-math-17k-gad.parquet bash gad_oprd_distillation.sh
```

**Two gotchas:**
- The launchers default to `data.truncation=error`; an over-long `teacher_response` will crash. Either
  size `data.max_response_length` to cover teacher solutions, or pass `data.truncation=right` (note:
  right-truncation drops the trailing `\boxed{}` answer — prefer enlarging the length).
- GAD's reward is `D(y)`, **not** `ground_truth` (we skip the external reward_fn), but keep
  `reward_model.ground_truth` in the parquet — validation and the OPD/OPRD baselines need it.
  `build_teacher_response_parquet.py` **adds** a column; it does not overwrite the others.

## Run

```bash
# Combined OPRD + GAD (lightweight in-training low-rank bridge)
ACTOR_MODEL_PATH=/models/Qwen3-1.7B-Base REWARD_MODEL_PATH=/models/Qwen3-4B \
GAD_COEF=0.5 REP_DISTILLATION_COEF=1.0 REP_LOW_RANK=8 \
bash gad_oprd_distillation.sh

# Baselines
bash baseline_oprd_only.sh
bash baseline_opd.sh
```

## Debugging

Three levels, easiest first.

**Level 0 — pure algorithm logic (any machine, no GPU/ray needed):**
```bash
python3 -m pdb tests/test_gad_components.py
```
Single-step the Bradley-Terry discriminator loss and the last-token scoring mask. The `real-import`
test in that file also does a numeric parity check against the *actual* verl function — it SKIPs on a
bare CPU box and PASSES once the env is installed.

**Level 1 — import smoke (after env install):**
```bash
python -c "from verl.trainer.ppo.core_algos import compute_discriminator_loss; \
           from verl.workers.critic.dp_critic import DataParallelPPOCritic; print('import ok')"
```

**Level 2 — single-GPU end-to-end debug (the main one):**
```bash
bash smoke_debug.sh     # 1 GPU, ~0.5B models, resp_len=512, rollout.n=2, 3 steps
```
`smoke_debug.sh` presets `RAY_DEBUG_POST_MORTEM=1` + `HYDRA_FULL_ERROR=1`, so **any exception inside a
worker drops into pdb** (post-mortem) — the most useful way to debug a verl crash. Requires a training
parquet that already has the `teacher_response` column.

**Breakpoints & the Ray multiprocess caveat.** verl runs under Ray:
- Code in `ray_trainer.fit()` (incl. **our reward injection** `if use_gad_discriminator:`) runs in the
  **driver** process — ordinary `breakpoint()` / IDE breakpoints **hit**.
- Code in `dp_critic` / `dp_actor` runs in **Ray worker** subprocesses — main-process breakpoints **do not
  hit**. To break there: (1) rely on `RAY_DEBUG_POST_MORTEM=1` (already set), or (2) put `breakpoint()` in
  the code, run with `export RAY_DEBUG=legacy`, and attach from another terminal via `ray debug`.
- For the loss *math*, prefer Level 0 (identical logic, single-step on CPU), then confirm wiring at Level 2.

Suggested breakpoints (after applying the patch):

| Where | What to inspect |
|---|---|
| `ray_trainer.py` → `if use_gad_discriminator:` (compute_values block) | `D(y)` becomes `token_level_scores` |
| `dp_critic.py: update_critic` (`if use_gad:` branch) | `d_loss` / `d_acc`, teacher vs student scores |
| `dp_critic.py: _slice_response_values` | only the last real response token is nonzero |
| `dp_actor.py` (near `policy_loss = pg_loss`) | `gad_coef·PG + rep_distillation_coef·rep` add cleanly |

### Smoke/debug knobs exposed by the patch

`on_policy_distillation.sh` now honors these env vars (defaults unchanged when unset) so a single-GPU
run needs no editing, and forwards extra Hydra overrides via `"$@"`:

| env | default | smoke value |
|---|---|---|
| `N_GPUS_PER_NODE` | 8 | 1 |
| `MAX_RESP_LENGTH` | 16384 | 512 |
| `N_RESPONSES` (GRPO group) | 2 | 2 |
| `MINI_BATCH_SIZE` | 8 | 2 |
| `TEST_FREQ` / `SAVE_FREQ` | 2 / 200 | large (skip) |

Example one-off override (appended, so keys not already set by the script — e.g. step limit):
```bash
bash smoke_debug.sh trainer.total_training_steps=3   # already the default inside smoke_debug.sh
```

## Architecture (four coexisting workers, no collision)

| worker | role | gate |
|---|---|---|
| actor / rollout | student (generator) | — |
| `reward_model` | frozen white-box **teacher** → hidden states for MSE | `reward_model.enable=True` |
| `critic` | **discriminator** (Bradley-Terry, → `D(y)` reward) | `critic.enable=True` + `use_gad_discriminator=True` |
| `ref` | KL anchor | `actor.use_kl_loss=True` |

## Source changes (in `oprd_gad.patch`)

- `core_algos.py` — `compute_discriminator_loss` (Bradley-Terry `-logσ(D(y_t)-D(y_s))`).
- `workers/config/actor.py`, `workers/config/critic.py`, `config/critic/critic.yaml` — `use_gad_discriminator`, `gad_coef`, `gad_gate_pg`.
- `workers/critic/dp_critic.py` — discriminator mode: `_forward_micro_batch(compute_teacher=)`, last-real-token score (`_slice_response_values`), BT `update_critic`. GAE value path preserved.
- `utils/dataset/rl_dataset.py` — optional `teacher_response` tokenization (guarded).
- `workers/rollout/vllm_rollout/vllm_rollout_spmd.py` — build `teacher_input_ids/attention_mask/position_ids` (guarded).
- `trainer/ppo/ray_trainer.py` — pop `teacher_response` into the gen batch; under GAD, `D(y_student)` → `token_level_scores` → GRPO advantage (external reward_fn skipped).
- `workers/actor/dp_actor.py` — scale the PG term by `gad_coef` (rep MSE term unchanged).
- `on_policy_distillation.sh` — `GAD_ARGS` block (off by default → baselines unchanged); exposes `N_GPUS_PER_NODE / MAX_RESP_LENGTH / N_RESPONSES / TEST_FREQ / SAVE_FREQ` as env vars (defaults unchanged) and forwards `"$@"` to the python call for ad-hoc Hydra overrides.

All GAD behavior is gated behind `use_gad_discriminator` + `adv_estimator=grpo`; OPRD-only and OPD are byte-for-byte unchanged when GAD is off.

## Verification status

- **Done (CPU/static):** `python3 tests/test_gad_components.py` passes (BT loss value/gradient, last-token masking); all edited `.py` compile; `bash -n` on all scripts; patch applies cleanly on base `93816fd`.
- **Requires GPU (not yet run — no GPU on the authoring machine):** FSDP co-sharding of actor + teacher(RM) + discriminator(critic); vLLM rollout with teacher sequences; end-to-end reward → advantage → PG + rep MSE; OOM behavior. **First GPU gate:** `bash scripts/smoke_debug.sh` (≈0.5B models, `rollout.n=2`, 3 steps) before the real 4B→1.5B run.

## Tuning / sharp edges (see plan for detail)

- **λ (`gad_coef`) start small (0.1–1)**; keep the rep term dominant. Watch `actor/rep_loss` vs `actor/pg_loss` and `critic/d_acc`.
- **Discriminator cold-start:** use `trainer.critic_warmup` and/or ramp `gad_coef` from 0; the MSE term carries early learning.
- **Memory:** three big forwards/step (teacher + discriminator×2 + actor). Keep the discriminator small; `param_offload=True` on RM/critic; `critic.use_dynamic_bsz=False`.
- **`teacher_response` length:** size `data.max_response_length` to cover teacher solutions (right-truncation drops the tail / final answer).
- **Reward lags one step** (discriminator scores before its own update) — expected, matches GAD.
- **Lightweight vs frozen bridge:** default uses in-training PCA `P_T` + joint `P_S` (`rep_freeze_ps=False`). For the stronger frozen bridge, build `ps_bank.pt` offline and set `REP_LOW_RANK_INIT_CHECKPOINT` + `rep_freeze_ps=True`.
