#!/bin/bash
# ==============================================================================
# Tiny single-GPU SMOKE + DEBUG run for OPRD+GAD (~3 steps).
# Exercises the full wiring end-to-end with post-mortem pdb on any worker exception.
# Requires a training parquet WITH a `teacher_response` column. NOT for real training.
# All knobs env-overridable (tiny defaults preserved). Copy next to on_policy_distillation.sh.
# ------------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- tiny footprint (override via env) ----
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-1}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-512}
export MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-512}
export N_RESPONSES=${N_RESPONSES:-2}            # GRPO group size
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-2}    # -> train_batch_size = 2 (PARALLEL_SIZE=1)
export TEST_FREQ=${TEST_FREQ:-100000}           # skip periodic validation during the smoke
export SAVE_FREQ=${SAVE_FREQ:-100000}           # skip checkpointing

# ---- SMALL models (EDIT to the smallest you have; teacher != student is fine for cross-arch) ----
export MODEL_DIR=${MODEL_DIR:-/workspace/model}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen2.5-0.5B-Instruct}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen2.5-0.5B-Instruct}
export DISCRIMINATOR_MODEL_PATH=${DISCRIMINATOR_MODEL_PATH:-${ACTOR_MODEL_PATH}}
export REP_LOW_RANK=${REP_LOW_RANK:-8}
export GAD_COEF=${GAD_COEF:-0.5}

# ---- debugger: drop into pdb at the point of ANY exception inside a Ray worker ----
export RAY_DEBUG_POST_MORTEM=${RAY_DEBUG_POST_MORTEM:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

# total_training_steps passed via the forwarded "$@" (not set inside the launcher -> no dup key)
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-3}
exec bash "$SCRIPT_DIR/gad_oprd_distillation.sh" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"
