#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/wby/llm-cache/nvidia-kvcache
PYTHON_BIN=/usr/bin/python3.12

if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  uv venv --python "${PYTHON_BIN}" --system-site-packages "${PROJECT_ROOT}/.venv"
fi

uv pip install --python "${PROJECT_ROOT}/.venv/bin/python" --no-deps \
  transformers==4.51.0 tokenizers==0.21.4 huggingface-hub==0.30.2 \
  datasets==2.19.0 accelerate==0.29.3 dill==0.3.8 multiprocess==0.70.16 \
  fsspec==2024.3.1 pyarrow-hotfix==0.7 responses==0.18.0 \
  sentencepiece==0.2.1 protobuf==6.32.0 xxhash==3.5.0 \
  typing_extensions==4.16.0

uv pip install --python "${PROJECT_ROOT}/.venv/bin/python" --no-deps -e "${PROJECT_ROOT}"
