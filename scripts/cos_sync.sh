#!/usr/bin/env zsh
# Sync the LMOps repo checkout with its COS backup (bucket from ~/.cos.conf, prefix
# `LMOps/` by default). Bridges this GPU box <-> the CPU box that pushes to git.
# Two directions:
#
#   (default)  PULL : COS   -> local   (download & merge into the working tree)
#   --push     PUSH : local -> COS      (upload the working tree to the backup)
#
# Typical flow:
#   GPU box  : scripts/cos_sync.sh --push        # publish local edits to COS
#   CPU box  : scripts/cos_sync.sh               # pull from COS, then `git add/commit/push`
#
# Safe by default:
#   * one direction per run;
#   * EXCLUDES .git/ (git lives on the CPU box; don't clobber it), the per-machine
#     bootstrap checkout `.oprd/`, training artifacts (outputs/ runs/ wandb/ *.pt *.ckpt
#     *.safetensors *.parquet *.tar.gz *.whl), datasets/, __pycache__/, *.pyc, *.log;
#   * incremental (coscmd -s / rsync --checksum skip unchanged files);
#   * --delete is OFF by default (never removes files missing on the source side).
#
# The public COS endpoint is unreachable from these boxes, so we force the INTERNAL
# endpoint via a TEMP copy of ~/.cos.conf — your ~/.cos.conf is never modified.
#
# Usage:
#   scripts/cos_sync.sh            [pull opts]     # COS -> local
#   scripts/cos_sync.sh --push     [push opts]     # local -> COS
#   common opts: --dry-run --delete --include-git
#                --prefix LMOps/ --endpoint HOST --env hf_download
set -e

# -------- defaults (overridable by flags or env) --------
COS_PREFIX="${COS_PREFIX:-LMOps/}"
COS_ENDPOINT="${COS_ENDPOINT:-cos-internal.ap-shanghai.tencentcos.cn}"  # internal; public is blocked here
CONDA_ENV="${CONDA_ENV:-hf_download}"                                   # env that has coscmd
MODE="pull"
DRY_RUN=0
DO_DELETE=0
INCLUDE_GIT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push) MODE="push"; shift;;
    --pull) MODE="pull"; shift;;
    --dry-run) DRY_RUN=1; shift;;
    --delete) DO_DELETE=1; shift;;
    --include-git) INCLUDE_GIT=1; shift;;
    --prefix) COS_PREFIX="$2"; shift 2;;
    --endpoint) COS_ENDPOINT="$2"; shift 2;;
    --env) CONDA_ENV="$2"; shift 2;;
    -h|--help) sed -n '2,29p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ "$COS_PREFIX" == */ ]] || COS_PREFIX="$COS_PREFIX/"

SCRIPT_DIR="${0:A:h}"                 # zsh: absolute dir of this script
REPO_ROOT="${SCRIPT_DIR:h}"           # parent of scripts/  -> the LMOps repo root
STAGE="$(mktemp -d /tmp/cos_sync.XXXXXX)"
TMP_CONF="$(mktemp /tmp/cos_sync_conf.XXXXXX)"
cleanup() { rm -rf "$STAGE" "$TMP_CONF"; }
trap cleanup EXIT INT TERM

# -------- coscmd (from the conda env if present) --------
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi
command -v coscmd >/dev/null 2>&1 || { echo "coscmd not found (try: conda activate $CONDA_ENV)" >&2; exit 3; }
[[ -f "$HOME/.cos.conf" ]] || { echo "~/.cos.conf not found — configure coscmd first (coscmd config ...)" >&2; exit 3; }

# -------- temp cos config with the internal endpoint (don't touch ~/.cos.conf) --------
cp "$HOME/.cos.conf" "$TMP_CONF"; chmod 600 "$TMP_CONF"
if [[ -n "$COS_ENDPOINT" ]] && ! grep -q '^endpoint' "$TMP_CONF"; then
  print "endpoint = $COS_ENDPOINT" >> "$TMP_CONF"
fi

# shared exclude list (rsync form) — never ship git, the per-machine .oprd checkout, or heavy artifacts
EXCLUDES=(
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '*.log'
  --exclude '.oprd/'                      # bootstrap's per-machine OPRD checkout (huge: outputs/ckpts)
  --exclude 'outputs/' --exclude 'runs/' --exclude 'wandb/' --exclude 'swanlab_log/'
  --exclude 'datasets/'
  --exclude '*.ckpt' --exclude '*.pt' --exclude '*.safetensors'
  --exclude '*.parquet' --exclude '*.tar.gz' --exclude '*.whl'
)
[[ "$INCLUDE_GIT" -eq 1 ]] || EXCLUDES+=(--exclude '.git/')

echo "[cos_sync] mode   : $MODE  (dry_run=$DRY_RUN delete=$DO_DELETE include_git=$INCLUDE_GIT)"
echo "[cos_sync] repo   : $REPO_ROOT"
echo "[cos_sync] cos    : cos://<bucket>/$COS_PREFIX  (endpoint=${COS_ENDPOINT:-<from ~/.cos.conf>})"

if [[ "$MODE" == "pull" ]]; then
  # COS -> staging (coscmd strips the prefix: LMOps/x -> $STAGE/x)
  coscmd -c "$TMP_CONF" download -rs "$COS_PREFIX" "$STAGE/"
  # staging -> repo (--checksum: coscmd resets mtimes; compare by content)
  RSYNC_ARGS=(-a --checksum --itemize-changes "${EXCLUDES[@]}")
  [[ "$DRY_RUN" -eq 1 ]] && RSYNC_ARGS+=(-n)
  [[ "$DO_DELETE" -eq 1 ]] && RSYNC_ARGS+=(--delete)
  echo "[cos_sync] rsync $STAGE/ -> $REPO_ROOT/"
  rsync "${RSYNC_ARGS[@]}" "$STAGE/" "$REPO_ROOT/"
  echo "  NOTE: .oprd / outputs / model weights are NOT synced — rebuild per machine via oprd_gad_delivery/bootstrap_oprd_gad.sh."
else
  # local -> staging (apply excludes so git/.oprd/artifacts never leave the machine)
  rsync -a "${EXCLUDES[@]}" "$REPO_ROOT/" "$STAGE/"
  echo "[cos_sync] staged $(find "$STAGE" -type f | wc -l | tr -d ' ') files for upload"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[cos_sync] (dry-run) would: coscmd upload -rs $([[ $DO_DELETE -eq 1 ]] && echo '--delete ')$STAGE/ -> $COS_PREFIX"
    (cd "$STAGE" && find . -type f | sed 's#^\./#  #') | head -60
  else
    UP_ARGS=(-rs)
    [[ "$DO_DELETE" -eq 1 ]] && UP_ARGS+=(--delete)   # mirror-delete remote files missing locally
    echo "[cos_sync] coscmd upload ${UP_ARGS[*]} $STAGE/ -> $COS_PREFIX"
    coscmd -c "$TMP_CONF" upload "${UP_ARGS[@]}" "$STAGE/" "$COS_PREFIX"
  fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[cos_sync] done (dry-run — nothing written)."
else
  echo "[cos_sync] done."
fi
