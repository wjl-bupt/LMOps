#!/bin/bash
# ==============================================================================
# bootstrap_oprd_gad.sh — one-shot setup + run for the OPRD+GAD merge.
#
# From this delivery folder, it:
#   1. clone   : clone the OPRD fork and pin it to the exact base commit (from ./BASE_COMMIT)
#   2. patch   : overlay our merge onto that checkout (robust file-copy; bypasses CRLF/patch issues)
#                + copy the launcher scripts into the repo
#   3. env     : create the conda env and install the verl stack (vllm/torch/flash-attn/ray/...)
#   4. check   : verify torch+CUDA on this box, import vllm/flash_attn, run CPU logic tests
#   5. run     : launch a training script (default: smoke_debug.sh) — NOT in the default set
#
# Idempotent & stage-selectable. Safe to re-run.
#
# Usage:
#   bash bootstrap_oprd_gad.sh                 # runs clone patch env check  (NOT run)
#   bash bootstrap_oprd_gad.sh clone patch     # only those stages
#   bash bootstrap_oprd_gad.sh all             # everything incl. run
#   RUN_SCRIPT=smoke_debug.sh bash bootstrap_oprd_gad.sh run
#   RUN_SCRIPT=gad_oprd_distillation.sh bash bootstrap_oprd_gad.sh run
#
# Config (override via env):
#   TARGET_DIR      where to clone OPRD          (default: $HOME/OPRD_gad)
#   OPRD_REPO_URL   OPRD git url                 (default: github ShenzhiYang2000/OPRD)
#   BASE_COMMIT     base commit to pin           (default: contents of ./BASE_COMMIT)
#   CONDA_ENV       conda env name               (default: verl)
#   PATCH_MODE      overlay | apply              (default: overlay — the reliable path)
#   RUN_SCRIPT      launcher for the run stage   (default: smoke_debug.sh)
#   RUN_ARGS        extra args to the launcher   (default: empty)
# ==============================================================================
set -euo pipefail

DELIVERY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPRD_REPO_URL="${OPRD_REPO_URL:-https://github.com/ShenzhiYang2000/OPRD.git}"
BASE_COMMIT="${BASE_COMMIT:-$(cat "$DELIVERY_DIR/BASE_COMMIT" 2>/dev/null || echo 93816fd)}"
TARGET_DIR="${TARGET_DIR:-$HOME/OPRD_gad}"
CONDA_ENV="${CONDA_ENV:-verl}"
PATCH_MODE="${PATCH_MODE:-overlay}"
RUN_SCRIPT="${RUN_SCRIPT:-smoke_debug.sh}"
RUN_ARGS="${RUN_ARGS:-}"
EXPECTED_CHANGED_FILES=10

log() { echo -e "\n\033[1;34m[bootstrap]\033[0m $*"; }
die() { echo -e "\033[1;31m[bootstrap:ERROR]\033[0m $*" >&2; exit 1; }

# stage selection: no args -> clone patch env check ; else the named stages ; 'all' -> everything
if [ $# -eq 0 ]; then REQUESTED=(clone patch env check); else REQUESTED=("$@"); fi
want() { local s; for s in "${REQUESTED[@]}"; do { [ "$s" = all ] || [ "$s" = "$1" ]; } && return 0; done; return 1; }

activate_conda() {
    command -v conda >/dev/null 2>&1 || die "conda not found on PATH. Install miniconda/anaconda first."
    # conda hook scripts trip 'set -u'; relax around them.
    set +u
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    set -u
}

# ------------------------------------------------------------------ 1. clone
stage_clone() {
    log "clone: $OPRD_REPO_URL  ->  $TARGET_DIR   (pin $BASE_COMMIT)"
    if [ -d "$TARGET_DIR/.git" ]; then
        log "repo already exists; fetching to ensure base commit is present"
        git -C "$TARGET_DIR" fetch --all --tags --quiet || true
    else
        # full clone (not --depth 1) so the exact base commit is guaranteed to be present
        git clone "$OPRD_REPO_URL" "$TARGET_DIR"
    fi
    git -C "$TARGET_DIR" checkout "$BASE_COMMIT"
    log "at commit: $(git -C "$TARGET_DIR" rev-parse HEAD)"
}

# ------------------------------------------------------------------ 2. patch
stage_patch() {
    [ -d "$TARGET_DIR/.git" ] || die "no repo at $TARGET_DIR (run the clone stage first)"
    if [ "$PATCH_MODE" = apply ]; then
        log "patch: git apply oprd_gad.patch (PATCH_MODE=apply)"
        git -C "$TARGET_DIR" apply "$DELIVERY_DIR/oprd_gad.patch" \
            || git -C "$TARGET_DIR" apply --3way "$DELIVERY_DIR/oprd_gad.patch" \
            || die "git apply failed; retry with PATCH_MODE=overlay"
    else
        log "patch: overlaying modified_files_full/ onto the checkout (robust, CRLF-proof)"
        [ -d "$DELIVERY_DIR/modified_files_full" ] || die "modified_files_full/ missing in delivery"
        cp -r "$DELIVERY_DIR/modified_files_full/." "$TARGET_DIR/"
    fi
    log "copying launcher scripts into the repo root"
    cp "$DELIVERY_DIR"/scripts/*.sh "$TARGET_DIR/"
    log "verification — git diff --stat (expect $EXPECTED_CHANGED_FILES files, +260/-29):"
    git -C "$TARGET_DIR" --no-pager diff --stat
    local n; n=$(git -C "$TARGET_DIR" diff --name-only | wc -l | tr -d ' ')
    if [ "$n" -ne "$EXPECTED_CHANGED_FILES" ]; then
        echo -e "\033[1;33m[bootstrap:WARN]\033[0m expected $EXPECTED_CHANGED_FILES changed files, got $n — check base commit / working tree."
    else
        log "OK: exactly $EXPECTED_CHANGED_FILES files changed — merge is in place."
    fi
}

# ------------------------------------------------------------------ 3. env
stage_env() {
    command -v conda >/dev/null 2>&1 || die "conda not found on PATH."
    set +u; source "$(conda info --base)/etc/profile.d/conda.sh"; set -u
    if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
        log "conda env '$CONDA_ENV' already exists — skipping create"
    else
        log "creating conda env '$CONDA_ENV' (python 3.12)"
        conda create -n "$CONDA_ENV" python==3.12 -y
    fi
    set +u; conda activate "$CONDA_ENV"; set -u
    log "installing verl stack (USE_MEGATRON=0): vllm0.11 / torch2.8 / flash-attn2.8.1 / flashinfer / ray ..."
    ( cd "$TARGET_DIR/verl" && USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh )
    pip install math-verify
    log "editable install of verl (so 'python -m verl.trainer.main_ppo' resolves)"
    ( cd "$TARGET_DIR/verl" && pip install -e . --no-deps )
    log "env ready. NOTE: nvidia-smi 'CUDA 12.4' is the driver max; torch 2.8 runs on it via CUDA 12.x minor-version compatibility."
}

# ------------------------------------------------------------------ 4. check
stage_check() {
    activate_conda
    log "check: torch + CUDA on this machine"
    python - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| built-cuda", torch.version.cuda,
      "| cuda_available", ok, "| device", torch.cuda.get_device_name(0) if ok else "N/A")
assert ok, "CUDA not available — check driver / torch install"
PY
    log "check: import vllm / flash_attn"
    python -c "import flash_attn, vllm; print('flash_attn', flash_attn.__version__, '| vllm', vllm.__version__)"
    log "check: CPU logic unit tests (BT loss + last-token masking + real-import parity)"
    python "$DELIVERY_DIR/tests/test_gad_components.py"
    log "all checks passed."
}

# ------------------------------------------------------------------ 5. run
stage_run() {
    [ -f "$TARGET_DIR/$RUN_SCRIPT" ] || die "launcher $RUN_SCRIPT not found in $TARGET_DIR (run the patch stage first)"
    activate_conda
    log "run: bash $RUN_SCRIPT $RUN_ARGS   (cwd: $TARGET_DIR)"
    echo -e "\033[1;33m[bootstrap:NOTE]\033[0m the training parquet must contain a 'teacher_response' column"
    echo -e "                (see scripts/build_teacher_response_parquet.py). Set MODEL paths via env or the launcher."
    ( cd "$TARGET_DIR" && bash "$RUN_SCRIPT" $RUN_ARGS )
}

# ------------------------------------------------------------------ orchestrate
log "delivery : $DELIVERY_DIR"
log "target   : $TARGET_DIR"
log "base     : $BASE_COMMIT"
log "stages   : ${REQUESTED[*]}"

want clone && stage_clone
want patch && stage_patch
want env   && stage_env
want check && stage_check
want run   && stage_run

log "done: [${REQUESTED[*]}]"
if ! want run; then
    echo -e "\nNext: prepare data (teacher_response column), then run a smoke test:\n  RUN_SCRIPT=smoke_debug.sh bash $(basename "${BASH_SOURCE[0]}") run"
fi
