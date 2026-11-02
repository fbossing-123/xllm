#!/usr/bin/env bash
# Staged A3 round: build, functional smoke, reuse soak and official-protocol
# performance probes. Every stage is idempotent and writes its own log under
# <run_dir>, so a short-lived jump authorization can resume where the previous
# one stopped.
#
#   bash tools/remote_round.sh <project_dir> <task_dir> <run_dir> <stages>
#
# stages: comma separated subset of prepare,accuracy,status,tests,perf_auto,
#         perf_off,perf_force,summary,all

set -euo pipefail

PROJECT="${1:?project dir}"
TASK="${2:?task data dir}"
RUN="${3:?output dir}"
STAGES="${4:-all}"

CANN_ENV="/usr/local/Ascend/ascend-toolkit/set_env.sh"
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif command -v python > /dev/null 2>&1; then
  PY=python
else
  PY=python3
fi
SUBMIT_SOC="ascend910_93"
DIRECT_SOC="ascend910_9382"
OPP="$PROJECT/build/packages/vendors/dual_affine_layernorm_custom"

mkdir -p "$RUN"
source "$CANN_ENV"
cd "$PROJECT"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"

want() {
  case ",$STAGES," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

stage_prepare() {
  echo "===== prepare ====="
  uname -n; uname -m; date -Is; head -2 /etc/os-release
  timeout 60 npu-smi info 2>&1 | tail -12 || true
  if [ ! -f "$OPP/op_api/lib/libcust_opapi.so" ] || [ ! -d "$PROJECT/build" ]; then
    if ! bash build.sh --soc "$SUBMIT_SOC" -j 4 > "$RUN/build.log" 2>&1; then
      echo "BUILD_FAILED"; tail -40 "$RUN/build.log"; return 1
    fi
  fi
  echo "BUILD_READY $(ls -t build/*.run 2>/dev/null | head -1)"
  export DUAL_AFFINE_LAYERNORM_SOC="$DIRECT_SOC"
  export ASCEND_CUSTOM_OPP_PATH="$OPP"
  if ! "$PY" -c "import torch, torch_npu, dual_affine_layernorm as d; d.extension._load_extension()" > /dev/null 2>&1; then
    if ! "$PY" setup.py build_ext --inplace > "$RUN/extension_build.log" 2>&1; then
      echo "EXTENSION_BUILD_FAILED"; tail -40 "$RUN/extension_build.log"; return 1
    fi
  fi
  echo "EXTENSION_READY"
  "$PY" -c "import torch, torch_npu; print('torch', torch.__version__, 'torch_npu', torch_npu.__version__)"
}

ensure_ml_dtypes() {
  if ! "$PY" -c "import ml_dtypes" > /dev/null 2>&1; then
    mkdir -p "$PROJECT/.deps"
    export PYTHONPATH="$PROJECT/.deps:$PYTHONPATH"
    if ! "$PY" -m pip install --no-index --no-deps --target "$PROJECT/.deps" "$PROJECT"/wheels/ml_dtypes-*.whl > "$RUN/ml_dtypes.log" 2>&1; then
      "$PY" -m pip install --no-cache-dir --no-deps --target "$PROJECT/.deps" "ml_dtypes==0.6.0" >> "$RUN/ml_dtypes.log" 2>&1 || echo "ML_DTYPES_INSTALL_FAILED"
    fi
  fi
  "$PY" -c "import ml_dtypes, numpy; print('ml_dtypes', ml_dtypes.__version__, 'numpy', numpy.__version__)" || echo "ML_DTYPES_UNAVAILABLE"
}

stage_accuracy() {
  echo "===== accuracy (46 public cases) ====="
  ensure_ml_dtypes
  if "$PY" "$TASK/verify.py" --solution-file "$PROJECT/solution.py" --device 0 \
      --report "$RUN/public_accuracy.json" > "$RUN/public_accuracy.log" 2>&1; then
    echo "ACCURACY_PASSED $(grep -c '"status": "passed"' "$RUN/public_accuracy.json" 2>/dev/null || echo 0)/46"
  else
    echo "ACCURACY_FAILED"; tail -20 "$RUN/public_accuracy.log"
  fi
}

stage_status() {
  echo "===== cache status ====="
  if ! "$PY" - <<'PYEOF' 2>&1 | tail -12
import json
import torch
import torch_npu
import dual_affine_layernorm as daln
print(json.dumps(dict(daln.cache_status()), indent=2))
PYEOF
  then
    echo "CACHE_STATUS_FAILED"
  fi
}

stage_tests() {
  echo "===== reuse cache and binding tests ====="
  if "$PY" -m pytest -q tests/test_npu_reuse_cache.py tests/test_npu_binding.py > "$RUN/npu_tests.log" 2>&1; then
    echo "NPU_TESTS_PASSED $(tail -1 "$RUN/npu_tests.log")"
  else
    echo "NPU_TESTS_FAILED"; tail -30 "$RUN/npu_tests.log"
  fi
}

run_probe() {
  local mode="$1"
  local entries="$2"
  local soak="$3"
  echo "===== performance: $mode ====="
  ensure_ml_dtypes
  local args=(tools/perf_probe.py --task-root "$TASK" --project "$PROJECT" --device 0
              --entries "$entries" --output "$RUN/perf_$mode.json")
  if [ -n "$soak" ]; then
    args+=(--soak "$soak")
  fi
  if DUAL_AFFINE_LAYERNORM_EXECUTOR_CACHE="$mode" "$PY" "${args[@]}" > "$RUN/perf_$mode.log" 2>&1; then
    echo "PERF_${mode}_OK"
  else
    echo "PERF_${mode}_FAILED"; tail -25 "$RUN/perf_$mode.log"
  fi
  grep -E '^\{' "$RUN/perf_$mode.log" || true
}

stage_summary() {
  echo "===== summary ====="
  "$PY" - "$RUN" <<'PYEOF' || true
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
for label in ("auto", "off", "force"):
    path = run / f"perf_{label}.json"
    if not path.is_file():
        print(f"{label}: missing")
        continue
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"{label}: unreadable ({error})")
        continue
    rows = []
    for case in report.get("cases", []):
        entry = case["entries"].get("run")
        if entry:
            rows.append((case["case_name"], entry["wall_us_median"], case["max_wall_us"]))
    passed = sum(1 for _, wall, limit in rows if wall <= limit)
    print(f"{label}: {passed}/{len(rows)} meet target; " +
          ", ".join(f"{name.split('_')[1]}={wall:.2f}/{limit}" for name, wall, limit in rows))
PYEOF
}

if want all; then
  STAGES="prepare,accuracy,status,tests,perf_auto,perf_off,perf_force,summary"
fi

for stage in $(echo "$STAGES" | tr ',' ' '); do
  case "$stage" in
    prepare) stage_prepare ;;
    accuracy) stage_accuracy ;;
    status) stage_status ;;
    tests) stage_tests ;;
    perf_auto) run_probe auto run,run_direct 16 ;;
    perf_off) run_probe off run "" ;;
    perf_force) run_probe force run 8 ;;
    summary) stage_summary ;;
    *) echo "UNKNOWN_STAGE:$stage"; exit 2 ;;
  esac
  echo "STAGE_DONE:$stage"
done

echo "ROUND_DONE"
date -Is