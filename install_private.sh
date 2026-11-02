#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: bash install_private.sh /absolute/path/custom_opp.run /absolute/user/prefix" >&2
    exit 2
fi
PACKAGE=$1
PREFIX=$2
[[ -f "$PACKAGE" && -s "$PACKAGE" ]] || { echo "package missing or empty" >&2; exit 2; }
[[ "$PREFIX" = /* && "$PREFIX" != / ]] || { echo "prefix must be a non-root absolute path" >&2; exit 2; }
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    [[ "$PREFIX" != "$ASCEND_HOME_PATH" && "$PREFIX" != "$ASCEND_HOME_PATH"/* ]] || {
        echo "refusing to install into system CANN" >&2; exit 2;
    }
fi
VENDOR_DIR="$PREFIX/vendors/dual_affine_layernorm_custom"
[[ ! -e "$VENDOR_DIR" ]] || { echo "refusing to overwrite existing $VENDOR_DIR" >&2; exit 2; }
mkdir -p "$PREFIX"
bash "$PACKAGE" --quiet --install-path="$PREFIX"
[[ -d "$VENDOR_DIR" ]] || { echo "installed vendor directory not found" >&2; exit 1; }
printf 'export ASCEND_CUSTOM_OPP_PATH=%q${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}\n' "$VENDOR_DIR"
