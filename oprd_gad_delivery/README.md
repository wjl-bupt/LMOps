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

## Contents

| Path | What |
|---|---|
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
`TARGET_DIR` (default `$HOME/OPRD_gad`), `CONDA_ENV` (`verl`), `PATCH_MODE` (`overlay`|`apply`),
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

# if run bash go to wrong, you can try this. if the raise error is cause by environment .
# pip install scipy==1.15.3、pip install matplotlib

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

### Configuration & env overrides

**Every setting is env-overridable with sensible defaults** — no need to edit the scripts:
```bash
MODEL_DIR=/models ACTOR_MODEL_PATH=/models/Qwen3-1.7B-Base REWARD_MODEL_PATH=/models/Qwen3-4B \
TRAIN_DATASET=../datasets/dapo-math-17k-gad.parquet \
TEST_FILE='["../datasets/test_data/AIME24/test.parquet"]' \
N_GPUS_PER_NODE=8 GAD_COEF=0.5 REP_LOW_RANK=8 MAX_RESP_LENGTH=8192 \
bash gad_oprd_distillation.sh
```
Common knobs: **models** `MODEL_DIR / ACTOR_MODEL_PATH / REWARD_MODEL_PATH / DISCRIMINATOR_MODEL_PATH`;
**data** `TRAIN_DATASET / TEST_FILE / TEST_DATA_DIR`; **resources** `N_GPUS_PER_NODE / PARALLEL_SIZE (tp) /
GPU_MEMORY_UTILIZATION / MINI_BATCH_SIZE / N_RESPONSES / MAX_RESP_LENGTH / MAX_PROMPT_LENGTH`; **OPRD**
`REP_DISTILLATION_COEF / REP_DISTILLATION_LAYERS / REP_DISTILLATION_POSITIONS / REP_DISTILLATION_LAST_K /
REP_LOW_RANK / REP_PROJECTOR_MODE / REP_FREEZE_PS`; **GAD** `GAD_COEF / GAD_GATE_PG / CRITIC_LR / CRITIC_MICRO_BSZ`;
**quick test** `TOTAL_TRAINING_STEPS / TEST_FREQ / SAVE_FREQ`.

Each launcher fixes only its **identity switches** (`USE_GAD_DISCRIMINATOR`, `USE_REP_DISTILLATION`,
`REP_DISTILLATION_ONLY`, `ADV_ESTIMATOR`) — to switch method, pick a different launcher, not an env var.
Because everything else is env-driven, **unset stale vars (or use a fresh shell) when switching methods**.

> Fixed in this delivery: `on_policy_distillation.sh` previously **hard-set** `ADV_ESTIMATOR`, model paths,
> `TRAIN_DATASET`, etc., which silently clobbered wrapper/env values (e.g. the combined run would have
> reverted to `token_reward_direct` instead of `grpo`). These are now `${VAR:-default}`.

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
