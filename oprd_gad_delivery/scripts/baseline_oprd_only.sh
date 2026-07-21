#!/bin/bash
# ==============================================================================
# Baseline: OPRD-only (representation MSE, no adversarial reward, no OPD reverse-KL)
# Reproduces the paper's OPRD-Vanilla / OPRD-Bridge distillation. GAD is OFF -> the
# critic/discriminator is never instantiated. This is the primary comparison baseline.
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_DIR=${MODEL_DIR:-/dockerdata/junewluo/models}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B-Base}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}

export USE_REP_DISTILLATION=True
export REP_DISTILLATION_ONLY=True           # rep MSE only (skips the PG block entirely)
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}
export REP_DISTILLATION_POSITIONS=last_k
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-2000}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_PROJECTOR_MODE=low_rank
export REP_LOW_RANK=${REP_LOW_RANK:-8}
export REP_FREEZE_PS=False

export USE_GAD_DISCRIMINATOR=False          # <-- no discriminator
export LOG_PROB_TOP_K=0
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}

export PROJECT_NAME=${PROJECT_NAME:-OPRD_only_baseline}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-oprd-only}

exec bash "$SCRIPT_DIR/on_policy_distillation.sh" "$@"
