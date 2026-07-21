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
export MODEL_DIR=${MODEL_DIR:-/dockerdata/junewluo/models}
export STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-${MODEL_DIR}/Qwen3-4B}                  # student (actor)
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-/dockerdata/junewluo/Qwen3-32B}         # white-box teacher (hidden states + teacher_response)
export DISCRIMINATOR_MODEL_PATH=${DISCRIMINATOR_MODEL_PATH:-${STUDENT_MODEL_PATH}}      # discriminator backbone (small)
export MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}                                             # GAD loads 3 model roles -> bf16 by default to avoid OOM

# ---- OPRD (primary) ----
export USE_REP_DISTILLATION=True
export REP_DISTILLATION_ONLY=False          # <-- False so the adversarial PG term is added
export REP_DISTILLATION_COEF=${REP_DISTILLATION_COEF:-1.0}    # mu
export REP_DISTILLATION_POSITIONS=last_k
export REP_DISTILLATION_LAST_K=${REP_DISTILLATION_LAST_K:-2000}
export REP_DISTILLATION_LAYERS=${REP_DISTILLATION_LAYERS:-all}
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

# ==============================================================================
# Step 0 (auto): build the `teacher_response` column the GAD discriminator requires.
# The pipeline does NOT generate teacher responses online — it only READS this column
# from the training parquet (rl_dataset.py) and the discriminator (dp_critic.py) hard-
# depends on it. So we generate it here with $TEACHER_MODEL_PATH if the file is absent.
# ==============================================================================
export GAD_BASE_DATASET=${GAD_BASE_DATASET:-${DATA_ROOT:-/dockerdata/junewluo/datasets}/dapo-math-17k-dedup.parquet}      # source parquet (no teacher_response)
export GAD_TRAIN_DATASET=${GAD_TRAIN_DATASET:-${GAD_BASE_DATASET%.parquet}-gad.parquet}   # output (with teacher_response)
export TEACHER_GEN_N=${TEACHER_GEN_N:-2000}              # #prompts to generate/keep (0 = all rows)
export TEACHER_GEN_MAX_TOKENS=${TEACHER_GEN_MAX_TOKENS:-4096}
export TEACHER_GEN_TP=${TEACHER_GEN_TP:-4}               # vLLM tensor-parallel for the (large) teacher
export FORCE_REBUILD_TEACHER=${FORCE_REBUILD_TEACHER:-0}

if [ ! -f "$GAD_TRAIN_DATASET" ] || [ "$FORCE_REBUILD_TEACHER" = "1" ]; then
    echo "[GAD step0] building teacher_response -> $GAD_TRAIN_DATASET"
    echo "[GAD step0]   teacher=$TEACHER_MODEL_PATH  base=$GAD_BASE_DATASET  n=$TEACHER_GEN_N  tp=$TEACHER_GEN_TP"
    python3 "$SCRIPT_DIR/build_teacher_response_parquet.py" \
        --in  "$GAD_BASE_DATASET" \
        --out "$GAD_TRAIN_DATASET" \
        --teacher "$TEACHER_MODEL_PATH" \
        --n "$TEACHER_GEN_N" \
        --max-tokens "$TEACHER_GEN_MAX_TOKENS" \
        --tp "$TEACHER_GEN_TP"
else
    echo "[GAD step0] teacher_response parquet already exists, skip build: $GAD_TRAIN_DATASET"
    echo "[GAD step0]   (set FORCE_REBUILD_TEACHER=1 to regenerate)"
fi

# Inject model paths + generated train data as Hydra overrides. These beat on_policy's
# hardcoded ACTOR/REWARD paths (env can't override those — they are reassigned there).
# User-supplied "$@" comes last, so it still wins over these defaults.
exec bash "$SCRIPT_DIR/on_policy_distillation.sh" \
    actor_rollout_ref.model.path="$STUDENT_MODEL_PATH" \
    reward_model.model.path="$TEACHER_MODEL_PATH" \
    critic.model.path="$DISCRIMINATOR_MODEL_PATH" \
    reward_model.model.fsdp_config.param_offload="${TEACHER_PARAM_OFFLOAD:-True}" \
    data.train_files="$GAD_TRAIN_DATASET" \
    "$@"
