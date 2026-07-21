#!/bin/bash
# ==============================================================================
# Baseline: OPRD-only (representation MSE, no adversarial reward, no OPD reverse-KL)
# Reproduces the paper's OPRD-Vanilla / OPRD-Bridge distillation. GAD is OFF -> the
# critic/discriminator is never instantiated. This is the primary comparison baseline.
#
# Every knob is env-overridable; the 4 IDENTITY switches (marked) are fixed.
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- identity (fixed) ----
export USE_REP_DISTILLATION=True            # [identity]
export REP_DISTILLATION_ONLY=True           # [identity] rep MSE only (skips the PG block entirely)
export USE_GAD_DISCRIMINATOR=False          # [identity] no discriminator
export ADV_ESTIMATOR=grpo                   # [identity]

# ---- models (override via env) ----
export MODEL_DIR=${MODEL_DIR:-/workspace/model}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B-Base}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}

# ---- OPRD hyperparams — all overridable ----
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-2000}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_PROJECTOR_MODE=${REP_PROJECTOR_MODE:-low_rank}
export REP_LOW_RANK=${REP_LOW_RANK:-8}
export REP_FREEZE_PS=${REP_FREEZE_PS:-False}

export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}
export USE_KL=${USE_KL:-True}

export PROJECT_NAME=${PROJECT_NAME:-OPRD_only_baseline}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-oprd-only}

exec bash "$SCRIPT_DIR/on_policy_distillation.sh" "$@"
