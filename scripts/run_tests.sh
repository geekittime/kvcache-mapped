#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/wby/llm-cache/nvidia-kvcache
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PROJECT_ROOT}/.venv/bin/python" -m pytest "$@"

