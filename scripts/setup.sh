#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${PYTHON_BIN:-python3.11}"
"$python_bin" - <<'PY'
import platform, sys, sysconfig, shutil
from pathlib import Path
assert sys.version_info[:2] == (3,11), 'Use Python 3.11; set PYTHON_BIN=/path/to/python3.11'
assert platform.system() == 'Linux' and platform.machine() == 'x86_64', 'This recipe targets Linux x86_64 with an NVIDIA GPU'
assert shutil.which('c++'), 'Install a C++ compiler (e.g. build-essential on Ubuntu) for runtime kernels'
assert (Path(sysconfig.get_path('include'))/'Python.h').exists(), 'Install development headers for this Python 3.11 interpreter'
if not shutil.which('nvcc'):
    print('CUDA compiler not on PATH. Install CUDA Toolkit 12.8 or matching FlashInfer prebuilt caches before the GPU doctor.')
PY
for phase in train infer; do
  "$python_bin" -m venv ".venv/$phase"
  ".venv/$phase/bin/python" -m pip install --upgrade 'pip==25.3'
done
# Separate processes and environments: never install vLLM into the training env.
.venv/train/bin/python -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
.venv/train/bin/python -m pip install -r requirements/train.lock.txt
.venv/infer/bin/python -m pip install 'torch==2.9.0' 'torchvision==0.24.0' 'torchaudio==2.9.0' --index-url https://download.pytorch.org/whl/cu128
.venv/infer/bin/python -m pip install -r requirements/infer.lock.txt
.venv/train/bin/python -m pip check
.venv/infer/bin/python -m pip check
.venv/train/bin/python scripts/doctor.py
echo 'Setup complete. Run .venv/train/bin/python scripts/doctor.py --gpu before training.'
