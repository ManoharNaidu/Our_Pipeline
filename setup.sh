#!/usr/bin/env bash
# One-step setup for Our_Pipeline: fetch the code (if needed), build the Python environment, check it, prepare data.
#
# Fastest start on a new machine (downloads only the code, not the old snapshot stored in the same repo):
#     curl -fsSLO https://huggingface.co/datasets/Manu2711/Our_Pipeline/resolve/main/setup.sh && bash setup.sh
# or, after downloading the dataset repo from Hugging Face, from inside that folder:
#     bash setup.sh
#
# Options:
#     --models          also pre-download the three models at the revisions pinned in configs/ (~35 GB)
#     --no-tests        skip the CPU test suite
#     --no-data         skip the data step
#     --download-only   only fetch the code, then stop
#     -h, --help        show this help
# Environment overrides:
#     REPO_ID      Hugging Face dataset repo with the code   (default Manu2711/Our_Pipeline)
#     REVISION     branch or commit to fetch                 (default main)
#     TORCH_INDEX  PyTorch wheel index                       (default https://download.pytorch.org/whl/cu130)
#     VENV         virtual environment folder                (default .venv)
#     DATA_ARCHIVE a .tar.gz holding data/ to unpack into the repo before the data checks
#     HF_TOKEN     token for a private repo or the gated Gemma model (or run `hf auth login`)
#     HF_HOME      where models are cached (Hugging Face default if unset)
#
# Safe to re-run: every step skips work that is already done. It never deletes anything; a broken environment
# folder is renamed to <VENV>.old-<time>.
set -euo pipefail

REPO_ID=${REPO_ID:-Manu2711/Our_Pipeline}
REVISION=${REVISION:-main}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
VENV=${VENV:-.venv}
MODELS=0; TESTS=1; DATA=1; DOWNLOAD_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --models) MODELS=1 ;;
    --no-tests) TESTS=0 ;;
    --no-data) DATA=0 ;;
    --download-only) DOWNLOAD_ONLY=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (see --help)"; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }
warn() { printf '\033[33mWARNING: %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*"; exit 1; }
NOTES=()

# ------------------------------------------------------------------ 1. code
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f "$HERE/src/our_pipeline/__init__.py" && -f "$HERE/pyproject.toml" ]]; then
  ROOT=$HERE
  say "1/6 Code: found in $ROOT"
else
  ROOT=$HERE/Our_Pipeline
  say "1/6 Code: downloading $REPO_ID@$REVISION into $ROOT"
  command -v curl >/dev/null || die "curl is needed to download the code"
  BASE="https://huggingface.co/datasets/$REPO_ID/resolve/$REVISION"
  AUTH=(); [[ -n "${HF_TOKEN:-}" ]] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
  mkdir -p "$ROOT"
  curl -fsSL "${AUTH[@]}" -o "$ROOT/code_manifest.txt" "$BASE/code_manifest.txt" \
    || die "could not fetch code_manifest.txt from $REPO_ID (private repo? set HF_TOKEN)"
  n=0
  while IFS= read -r path; do
    [[ -z "$path" || "$path" == \#* ]] && continue
    [[ -f "$ROOT/$path" ]] && continue
    curl -fsSL "${AUTH[@]}" --create-dirs -o "$ROOT/$path" "$BASE/$path" || die "download failed: $path"
    n=$((n + 1))
  done < "$ROOT/code_manifest.txt"
  echo "downloaded $n files"
  chmod +x "$ROOT/setup.sh" 2>/dev/null || true
fi
cd "$ROOT"
[[ $DOWNLOAD_ONLY == 1 ]] && { echo "Code is in $ROOT. Next: bash $ROOT/setup.sh"; exit 0; }

# ------------------------------------------------------------------ 2. Python environment
say "2/6 Python environment ($VENV)"
if ! command -v uv >/dev/null; then
  if command -v python3 >/dev/null && python3 -m pip --version >/dev/null 2>&1; then
    echo "installing uv with pip (user install)"
    python3 -m pip install --user --quiet uv && export PATH="$HOME/.local/bin:$PATH"
  fi
fi
command -v uv >/dev/null || die "uv is needed: see https://docs.astral.sh/uv/getting-started/installation/"
if [[ -d "$VENV" ]] && ! "$VENV/bin/python" -c 'import sys; assert sys.version_info >= (3, 12)' 2>/dev/null; then
  old="$VENV.old-$(date +%Y%m%d-%H%M%S)"
  warn "$VENV exists but has no working Python 3.12 (e.g. copied from another machine); renaming it to $old"
  mv "$VENV" "$old"
fi
[[ -d "$VENV" ]] || uv venv --quiet --python 3.12 "$VENV"
PY="$ROOT/$VENV/bin/python"
"$PY" --version

if command -v nvidia-smi >/dev/null; then
  # "CUDA Version: 12.8" on older drivers, "CUDA UMD Version: 13.4" on newer ones
  cuda=$(nvidia-smi | sed -n 's/.*CUDA[A-Z ]*Version: *\([0-9][0-9]*\.[0-9]*\).*/\1/p' | head -1)
  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
  echo "GPU: $gpu; driver supports CUDA ${cuda:-unknown}"
  if [[ -n "$cuda" && "${TORCH_INDEX##*/}" == cu13* && "${cuda%%.*}" -lt 13 ]]; then
    warn "the driver supports CUDA $cuda but TORCH_INDEX is ${TORCH_INDEX##*/}; set TORCH_INDEX to a build for CUDA <= $cuda"
  fi
else
  warn "nvidia-smi not found: tests run on CPU, but summarising and training need a GPU"
fi

if "$PY" -c 'import torch, transformers, peft, our_pipeline' 2>/dev/null; then
  echo "packages already installed"
else
  echo "installing torch 2.14.1 from $TORCH_INDEX"
  uv pip install --quiet --python "$PY" "torch==2.14.1" --index-url "$TORCH_INDEX"
  echo "installing our-pipeline (editable) with dev extras"
  uv pip install --quiet --python "$PY" -e ".[dev]"
fi
"$PY" -c 'import torch, transformers, peft; print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "| transformers", transformers.__version__, "| peft", peft.__version__)'

# ------------------------------------------------------------------ 3. tests
if [[ $TESTS == 1 ]]; then
  say "3/6 Tests (CPU)"
  "$PY" -m pytest -q -p no:cacheprovider 2>&1 | tail -3
else
  say "3/6 Tests: skipped (--no-tests)"
fi

# ------------------------------------------------------------------ 4. Hugging Face login and models
say "4/6 Hugging Face access"
HF="$ROOT/$VENV/bin/hf"
if [[ -n "${HF_TOKEN:-}" ]] || "$HF" auth whoami >/dev/null 2>&1; then
  "$HF" auth whoami 2>/dev/null | head -1 || true
  # asks the Hub only (no download, no cache), so a broken model cache cannot give a false "no access"
  if "$PY" -c 'from huggingface_hub import auth_check; auth_check("google/gemma-2-9b-it")' >/dev/null 2>&1; then
    echo "gemma-2-9b-it (gated): access OK"
  else
    warn "no access to google/gemma-2-9b-it: accept its licence on huggingface.co, then re-run"
    NOTES+=("Gemma-2-9B-it is gated: accept the licence on its Hugging Face page")
  fi
else
  warn "not logged in to Hugging Face: run  $HF auth login  (needed for the gated Gemma model), then re-run"
  NOTES+=("Log in: $HF auth login")
fi
if [[ $MODELS == 1 ]]; then
  echo "pre-downloading the pinned models (HF_HOME=${HF_HOME:-default})"
  # model and revision pairs read from the configs, so they cannot drift from what the pipeline loads
  "$PY" - <<'EOF' | while read -r model rev; do "$HF" download "$model" --revision "$rev" >/dev/null && echo "  $model@${rev:0:8} ok"; done
from pathlib import Path
from our_pipeline.config import load_config
pairs = set()
for path in sorted(Path("configs").glob("datasets/*.yaml")) + sorted(Path("configs").glob("ours/[a-z]*_k?.yaml")):
    cfg = load_config(path)
    for section in ("embedder", "summarizer", "classifier"):
        part = cfg.get(section) or {}
        if part.get("model_id") and part.get("revision"):
            pairs.add((part["model_id"], part["revision"]))
for model, rev in sorted(pairs):
    print(model, rev)
EOF
else
  echo "models download on first use (or re-run with --models to fetch them now, ~35 GB)"
fi

# ------------------------------------------------------------------ 5. data
if [[ $DATA == 1 ]]; then
  say "5/6 Data"
  if [[ -n "${DATA_ARCHIVE:-}" ]]; then
    [[ -f "$DATA_ARCHIVE" ]] || die "DATA_ARCHIVE not found: $DATA_ARCHIVE"
    echo "unpacking $DATA_ARCHIVE (existing files are kept)"
    tar -xzf "$DATA_ARCHIVE" --skip-old-files -C "$ROOT"
  fi
  # AmazonVideo: the raw file has a public URL in its config, so it can always be rebuilt
  av_raw=data/raw/amazonvideo/reviews_Amazon_Instant_Video_5.json.gz
  av_url=$(sed -n 's/^ *url: *//p' configs/datasets/amazonvideo.yaml | head -1)
  if [[ ! -f data/processed/amazonvideo/graph.json && ! -f "$av_raw" && -n "$av_url" ]]; then
    echo "downloading AmazonVideo raw reviews"
    curl -fsSL --create-dirs -o "$av_raw" "$av_url" || warn "AmazonVideo download failed: $av_url"
  fi
  for ds in amazonvideo yelpchi; do
    if [[ -f data/processed/$ds/graph.json && -f data/processed/$ds/split.json ]]; then
      echo "$ds: ready"
    elif [[ -f data/raw/$ds/$( [[ $ds == yelpchi ]] && echo reviews.jsonl || echo reviews_Amazon_Instant_Video_5.json.gz ) ]]; then
      echo "$ds: building from raw files"
      "$PY" scripts/prepare_data.py --config "configs/datasets/$ds.yaml" | tail -1
    else
      echo "$ds: MISSING"
      NOTES+=("$ds: put the raw file at the path in configs/datasets/$ds.yaml, then re-run")
    fi
  done
  for ds in instagram reddit amazon_musical; do
    if [[ -f data/processed/$ds/graph.json && -f data/processed/$ds/split.json ]]; then
      echo "$ds: ready"
    else
      echo "$ds: MISSING"
      NOTES+=("$ds: no public download is recorded; copy its processed graph into data/processed/$ds/ (with split.json)")
    fi
  done
else
  say "5/6 Data: skipped (--no-data)"
fi

# ------------------------------------------------------------------ 6. summary
say "6/6 Done"
echo "Repo:        $ROOT"
echo "Python:      $PY"
if ((${#NOTES[@]})); then
  echo "Still to do:"
  for n in "${NOTES[@]}"; do echo "  - $n"; done
fi
cat <<EOF
Next, for example:
  cd $ROOT
  $VENV/bin/python scripts/run_queue.py configs/ours/amazonvideo_k1.yaml --gpus 0
  $VENV/bin/python scripts/evaluate.py configs/ours/amazonvideo_k1.yaml
EOF
