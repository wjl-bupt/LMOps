#!/bin/bash
# ==============================================================================
# Baseline: OPD (output-space top-k reverse-KL distillation), optionally + OPRD.
# This is the native OPRD-fork OPD path, UNCHANGED by the GAD merge. GAD is OFF.
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_DIR=${MODEL_DIR:-/dockerdata/junewluo/models}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B-Base}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}

# OPD: top-k reverse KL on output distributions. Set REP off for pure OPD, or leave
# rep on for OPD+OPRD (the paper's L = L_OPD + mu*L_OPRD).
export USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-False}
export REP_DISTILLATION_ONLY=False
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}     # top-k for OPD reverse-KL
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}

export USE_GAD_DISCRIMINATOR=False              # <-- no discriminator
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}

export PROJECT_NAME=${PROJECT_NAME:-OPD_baseline}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-opd}

exec bash "$SCRIPT_DIR/on_policy_distillation.sh" "$@"
