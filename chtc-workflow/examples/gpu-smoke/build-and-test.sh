#!/bin/bash
# Run INSIDE a CHTC interactive build job, after files have transferred.
set -euo pipefail
: "${_CONDOR_SCRATCH_DIR:?Use condor_submit -i build-test.sub first}"
cd "$_CONDOR_SCRATCH_DIR"
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export APPTAINER_CACHEDIR="$PWD/.cache/apptainer"
export APPTAINER_TMPDIR="$PWD/.apptainer-tmp"
mkdir -p "$APPTAINER_TMPDIR"
chmod +x uv chtc
# Explicit uv venv -> torch installation -> export -> container.
./uv venv --python 3.11 .venv
./uv pip install --python .venv/bin/python torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda)'
export PATH="$PWD:$PATH"
./chtc build --project . --venv .venv --tag chtc-gpu-smoke:torch2.5.1-py311 \
    --prepare --base python:3.11-slim-bookworm --exclude uv --exclude chtc \
    --index-url https://download.pytorch.org/whl/cu124
context=$(find .chtc/builds -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
cd "$context"
export APPTAINER_CACHEDIR="$_CONDOR_SCRATCH_DIR/.cache/apptainer"
export APPTAINER_TMPDIR="$_CONDOR_SCRATCH_DIR/.apptainer-tmp"
apptainer build image.sif image.def
apptainer exec image.sif /opt/venv/bin/python -c \
    'import sys, torch; assert sys.version_info[:2] == (3,11); print(torch.__version__, torch.version.cuda)'
echo 'Container built and import checked.'
echo 'Copy image.sif to an approved persistent image location BEFORE exiting.'
echo 'Then use chtc submit --image IMAGE_PATH_OR_URL --gpus 1 smoke.py (see README).'
