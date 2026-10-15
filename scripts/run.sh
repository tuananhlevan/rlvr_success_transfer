#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONUNBUFFERED=1
export RLVR_TRAIN_PYTHON="$project_dir/.venv/train/bin/python"
export RLVR_INFER_PYTHON="$project_dir/.venv/infer/bin/python"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [[ ! -x "$RLVR_TRAIN_PYTHON" || ! -x "$RLVR_INFER_PYTHON" ]]; then
  echo 'Missing Python environments. Run bash scripts/setup.sh first.' >&2
  exit 1
fi
exec "$RLVR_TRAIN_PYTHON" -B -u studies/rlvr_success_transfer/reproduce_failure.py "$@"
