#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOC=ascend910_93
JOBS=4
while [[ $# -gt 0 ]]; do
    case "$1" in
        --soc) SOC=$2; shift 2 ;;
        -j|--jobs) JOBS=$2; shift 2 ;;
        -h|--help) printf 'Usage: bash build.sh [--soc ascend910_93] [-j 4]\n'; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
: "${ASCEND_HOME_PATH:?source the CANN set_env.sh first}"
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "jobs must be positive" >&2; exit 2; }
cmake -S "$ROOT" -B "$ROOT/build" \
    -DASCEND_COMPUTE_UNIT="$SOC" -DCANNJUDGE_COMPILE_JOBS="$JOBS"
for target in all binary package install; do
    cmake --build "$ROOT/build" --target "$target" --parallel "$JOBS"
done
shopt -s nullglob
packages=("$ROOT"/build/custom_opp_*.run)
[[ ${#packages[@]} -gt 0 && -s "${packages[0]}" ]] || { echo "RUN package missing" >&2; exit 1; }
printf 'PACKAGE=%s\n' "${packages[0]}"
printf 'VENDOR_DIR=%s/build/packages/vendors/dual_affine_layernorm_custom\n' "$ROOT"
