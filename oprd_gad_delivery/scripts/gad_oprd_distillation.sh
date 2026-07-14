#!/bin/bash
# ==============================================================================
# OPRD + GAD combined run (representation MSE primary + adversarial reward auxiliary)
# ==============================================================================
# Objective:  L_actor = gad_coef * PG(D(y_student))  +  rep_distillation_coef * MSE(h_student, h_teacher)  +  KL
#   - OPRD (teacher hidden-state MSE)   -> primary, deterministic, low-variance signal
#   - GAD  (discriminator reward, GRPO) -> auxiliary, global adversarial signal
#
# White-box, cross-architecture (big teacher -> small student). Uses the LIGHTWEIGHT bridge:
#   rep_projector_mode=low_rank + in-training PCA init of P_T + jointly-trained P_S
#   (no offline 3-stage bridge construction). See README for the frozen-bridge upgrade.
#
# Data contract: the training parquet MUST contain a `teacher_response` TEXT column
#   (the teacher's own solution to each prompt) — this is the discriminator's "real" example.
#
# Copy this file into the OPRD fork root (next to on_policy_distillation.sh) and run it.
# ------------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- models (EDIT) ----
export MODEL_DIR=${MODEL_DIR:-/workspace/model}
export ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-${MODEL_DIR}/Qwen3-1.7B-Base}   # student
export REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}        # white-box teacher (hidden states)
export DISCRIMINATOR_MODEL_PATH=${DISCRIMINATOR_MODEL_PATH:-${ACTOR_MODEL_PATH}}  # discriminator backbone (small)

# ---- OPRD (primary) ----
export USE_REP_DISTILLATION=True
export REP_DISTILLATION_ONLY=False          # <-- False so the adversarial PG term is added
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}    # mu
export REP_DISTILLATION_POSITIONS=last_k
export REP_DISTILLATION_LAST_K=2000
export REP_DISTILLATION_LAYERS=all
export REP_PROJECTOR_MODE=low_rank          # cross-arch bridge
export REP_LOW_RANK=${REP_LOW_RANK:-8}
export REP_FREEZE_PS=False                  # joint P_S (lightweight bridge)
# Optional frozen-bridge upgrade (stronger, per paper). Build ps_bank.pt offline first, then:
# export REP_LOW_RANK_INIT_CHECKPOINT=${SCRIPT_DIR}/outputs/bridge_construction/rank_${REP_LOW_RANK}/ps_bank.pt
# export REP_FREEZE_PS=True

# ---- GAD (auxiliary) ----
export USE_GAD_DISCRIMINATOR=True
export GAD_COEF=${GAD_COEF:-0.5}            # lambda: START SMALL (0.1-1). Keep rep term dominant.
export GAD_GATE_PG=True
export CRITIC_LR=${CRITIC_LR:-1e-6}
export CRITIC_MICRO_BSZ=${CRITIC_MICRO_BSZ:-1}

# ---- no OPD top-k reverse-KL in the primary method (ablation only) ----
export LOG_PROB_TOP_K=0
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export USE_KL=${USE_KL:-True}

export PROJECT_NAME=${PROJECT_NAME:-OPRD_GAD}
export TRAIN_DATASET_NAME=${TRAIN_DATASET_NAME:-DAPO-Math-17k-oprd-gad}

exec bash "$SCRIPT_DIR/on_policy_distillation.sh" "$@"
