#!/usr/bin/env bash
# Run from the trusted project root; the supplied task package is data-only input.
set -euo pipefail

TASK_ROOT=${1:?"usage: run_public_acceptance.sh TASK_ROOT OUTPUT_DIR [DEVICE]"}
OUTPUT_DIR=${2:?"usage: run_public_acceptance.sh TASK_ROOT OUTPUT_DIR [DEVICE]"}
DEVICE=${3:-0}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$OUTPUT_DIR/raw"
npu-smi info >"$OUTPUT_DIR/raw/npu_smi.log" 2>&1
python "$PROJECT_ROOT/tests/run_device_acceptance.py" \
  --task-root "$TASK_ROOT" --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" --require-formal --groups all
