#!/bin/bash
# ==============================================================================
# Baseline: OPD (output-space top-k reverse-KL distillation), optionally + OPRD.
# Native OPRD-fork OPD path, UNCHANGED by the GAD merge. GAD is OFF.
# NOTE: OPD needs teacher & student to share the SAME tokenizer/vocab (e.g. Qwen3 family).
#
# Every knob is env-overridable; the 3 IDENTITY switches (marked) are fixed.
# For OPD+OPRD, set USE_REP_DISTILLATION=True.
# ==============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- identity (fixed) ----
export REP_DISTILLATION_ONLY=False              # [identity] OPD always has the PG term
export USE_GAD_DISCRIMINATOR=False              # [identity] no discriminator
export ADV_ESTIMATOR=token_reward_direct        # [identity] OPD token-level reward

# ---- models (override via env) ----
export MODEL_DIR=${MODEL_DIR:-/workspace/model}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B-Base}
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}

# ---- OPD hyperparams — all overridable ----
export USE_REP_DISTILLATION=${USE_REP_DISTILLATION:-False}   # True -> OPD + OPRD (L = L_OPD + mu*L_OPRD)
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}                  # top-k for OPD reverse-KL
export TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
export REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
export USE_KL=${USE_KL:-True}

export PROJECT_NAME=${PROJECT_NAME:-OPD_baseline}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-opd}

exec bash "$SCRIPT_DIR/on_policy_distillation.sh" "$@"
