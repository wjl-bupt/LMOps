#!/bin/bash
# ==============================================================================
# Tiny single-GPU SMOKE + DEBUG run for OPRD+GAD.
# Goal: exercise the full wiring (rollout -> teacher hidden states -> discriminator D(y)
#       -> GRPO advantage + rep MSE -> actor/critic update) end-to-end in ~3 steps on 1 GPU,
#       with post-mortem pdb on any worker exception. NOT for real training.
#
# Requires: a training parquet WITH a `teacher_response` column (see build_teacher_response_parquet.py).
# Copy this next to on_policy_distillation.sh in the patched OPRD repo, then run it.
# ------------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- tiny footprint (env vars now honored by the patched on_policy_distillation.sh) ---
export N_GPUS_PER_NODE=1
export MAX_RESP_LENGTH=512
export MAX_VAL_RESP_LENGTH=512
export N_RESPONSES=2            # GRPO group size
export MINI_BATCH_SIZE=2        # -> train_batch_size = 2 (PARALLEL_SIZE=1)
export TEST_FREQ=100000         # skip periodic validation during the smoke
export SAVE_FREQ=100000         # skip checkpointing

# --- SMALL models (EDIT to the smallest you have; teacher != student is fine for cross-arch) ---
export MODEL_DIR=${MODEL_DIR:-/workspace/model}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen2.5-0.5B-Instruct}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen2.5-0.5B-Instruct}
export DISCRIMINATOR_MODEL_PATH=${DISCRIMINATOR_MODEL_PATH:-${ACTOR_MODEL_PATH}}
export REP_LOW_RANK=${REP_LOW_RANK:-8}
export GAD_COEF=${GAD_COEF:-0.5}

# --- debugger: drop into pdb at the point of ANY exception inside a Ray worker ---
export RAY_DEBUG_POST_MORTEM=1
export HYDRA_FULL_ERROR=1

# total_training_steps=3 is passed via the forwarded "$@" (the script does NOT set this key
# itself, so there is no Hydra duplicate-key error). Append more overrides after it if needed.
exec bash "$SCRIPT_DIR/gad_oprd_distillation.sh" \
    trainer.total_training_steps=3 \
    "$@"
