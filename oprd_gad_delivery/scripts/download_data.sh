#!/bin/bash
# ==============================================================================
# download_data.sh — fetch the verl-format datasets for OPRD+GAD.
#   train : DAPO-Math-17k     (HuggingFace, already verl-format)
#   eval  : AIME24            (HuggingFace, already verl-format)
# Both parquets are ready to use (columns: prompt / data_source / reward_model.ground_truth /
# ability / extra_info) — NO preprocessing needed. AIME24 is a *validation* set, not training data.
#
# Idempotent: skips files that already exist (OVERWRITE=1 to refetch).
# Paths match what the launchers expect:
#   on_policy_distillation.sh -> TRAIN_DATASET=../datasets/dapo-math-17k.parquet
#                                TEST_DATA_DIR=../datasets/test_data
# Run this from the OPRD repo root so `../datasets` resolves there, or set DATA_ROOT explicitly.
# ==============================================================================
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-../datasets}"
OVERWRITE="${OVERWRITE:-0}"

TRAIN_URL="https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
AIME24_URL="https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"

TRAIN_OUT="$DATA_ROOT/dapo-math-17k.parquet"
AIME24_OUT="$DATA_ROOT/test_data/AIME24/test.parquet"

fetch() {  # <url> <out>
    local url="$1" out="$2"
    mkdir -p "$(dirname "$out")"
    if [ -f "$out" ] && [ "$OVERWRITE" != "1" ]; then
        echo "[download] exists, skip: $out   (OVERWRITE=1 to refetch)"
        return
    fi
    echo "[download] -> $out"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$out" "$url"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "$out" "$url"
    else
        echo "ERROR: neither wget nor curl found" >&2; exit 1
    fi
}

echo "[download] DATA_ROOT=$DATA_ROOT"
fetch "$TRAIN_URL"  "$TRAIN_OUT"     # training  (DAPO-Math-17k)
fetch "$AIME24_URL" "$AIME24_OUT"    # eval      (AIME24 -> $TEST_DATA_DIR/AIME24/test.parquet)

cat <<EOF

[download] done.
  train : $TRAIN_OUT
  eval  : $AIME24_OUT
          use it via:  export TEST_FILE='["$DATA_ROOT/test_data/AIME24/test.parquet"]'

Next, for GAD, add the discriminator's teacher_response column to the TRAIN parquet:
  python3 scripts/build_teacher_response_parquet.py \\
      --in  $TRAIN_OUT \\
      --out ${TRAIN_OUT%.parquet}-gad.parquet \\
      --teacher \$REWARD_MODEL_PATH
  # then:  TRAIN_DATASET=${TRAIN_OUT%.parquet}-gad.parquet bash gad_oprd_distillation.sh

Other eval sets (AIME25 / AMC23 / MATH-500 / ...) live on HuggingFace in the same verl format;
put each at \$DATA_ROOT/test_data/<BENCH>/test.parquet and add it to TEST_FILE.
EOF
