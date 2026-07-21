#!/bin/bash
# ==============================================================================
# OPRD + GAD combined run (representation MSE primary + adversarial reward auxiliary)
# ==============================================================================
# Objective:  L_actor = gad_coef * PG(D(y_student))  +  rep_distillation_coef * MSE(h_student, h_teacher)  +  KL
#   - OPRD (teacher hidden-state MSE)   -> primary, deterministic, low-variance signal
#   - GAD  (discriminator reward, GRPO) -> auxiliary, global adversarial signal
#
# White-box, cross-architecture (big teacher -> small student). Uses the LIGHTWEIGHT bridge:
#   rep_projector_mode=low_rank + in-training PCA init of P_T + jointly-trained P_S.
#
# Data contract: the training parquet MUST contain a `teacher_response` TEXT column.
# Copy this file into the OPRD fork root (next to on_policy_distillation.sh) and run it.
#
# EVERY knob below is env-overridable: `GAD_COEF=1.0 REP_LOW_RANK=16 bash gad_oprd_distillation.sh`.
# The 4 IDENTITY switches (marked) are fixed — they define "this is the OPRD+GAD run".
# ------------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- identity (fixed): what makes this the combined OPRD+GAD run ----
export USE_REP_DISTILLATION=True            # [identity] rep MSE on
export REP_DISTILLATION_ONLY=False          # [identity] False -> the adversarial PG term is added
export USE_GAD_DISCRIMINATOR=True           # [identity] discriminator on
export ADV_ESTIMATOR=grpo                   # [identity] D(y) -> group-normalized advantage

# ---- models (override via env) ----
export MODEL_DIR=${MODEL_DIR:-/workspace/model}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B-Base}          # student
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}              # white-box teacher
export DISCRIMINATOR_MODEL_PATH=${DISCRIMINATOR_MODEL_PATH:-${ACTOR_MODEL_PATH}}  # discriminator backbone

# ---- OPRD (mu = rep_distillation_coef) — all overridable ----
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}
export REP_DISTILLATION_POSITIONS=${REP_DISTILLATION_POSITIONS:-last_k}
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-2000}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
export REP_PROJECTOR_MODE=${REP_PROJECTOR_MODE:-low_rank}     # cross-arch bridge
export REP_LOW_RANK=${REP_LOW_RANK:-8}
export REP_FREEZE_PS=${REP_FREEZE_PS:-False}                  # False = joint P_S (lightweight bridge)
# frozen-bridge upgrade: REP_LOW_RANK_INIT_CHECKPOINT=/path/ps_bank.pt REP_FREEZE_PS=True

# ---- GAD (lambda = gad_coef) — all overridable ----
export GAD_COEF=${GAD_COEF:-0.5}            # START SMALL (0.1-1); keep the rep term dominant
export GAD_GATE_PG=${GAD_GATE_PG:-True}
export CRITIC_LR=${CRITIC_LR:-1e-6}
export CRITIC_MICRO_BSZ=${CRITIC_MICRO_BSZ:-1}

# ---- OPD top-k reverse-KL: off in the primary method; set >0 for a three-way ablation ----
export LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-0}
export USE_KL=${USE_KL:-True}

export PROJECT_NAME=${PROJECT_NAME:-OPRD_GAD}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-oprd-gad}

exec bash "$SCRIPT_DIR/on_policy_distillation.sh" "$@"
