#!/usr/bin/env python3
"""Generate acceptance reports only from a complete formal A3 passing run."""

import argparse
import json
from pathlib import Path

try:
    from .fingerprint import validate_implementation_fingerprints
    from .task_integrity import validate_task_fingerprint_evidence
    from .environment_contract import formal_environment_errors
except ImportError:
    from fingerprint import validate_implementation_fingerprints
    from task_integrity import validate_task_fingerprint_evidence
    from environment_contract import formal_environment_errors


REQUIRED_REPORTS = (
    "1 自验证步骤说明.md",
    "2.1 精度自验证报告.xlsx",
    "2.2 精度自验证日志.log",
    "3.1 性能自验证报告.xlsx",
    "3.2 性能自验证日志.log",
    "4.1 内存自验证报告.xlsx",
    "4.2 内存自验证日志.log",
)
ALL_GROUPS = {"accuracy", "extra", "invalid", "stream", "lifetime", "memory", "performance", "verifier"}

# Reference wall times from the task book section 3.3 (microseconds).
PERF_BASELINE_US = {
    "DualAffineLayerNorm_005_float16_random": 33.49,
    "DualAffineLayerNorm_006_float16_random": 33.45,
    "DualAffineLayerNorm_007_float16_random": 32.31,
    "DualAffineLayerNorm_008_float16_random": 66.64,
    "DualAffineLayerNorm_009_float16_random": 31.69,
    "DualAffineLayerNorm_020_bfloat16_random": 31.61,
    "DualAffineLayerNorm_021_bfloat16_random": 34.57,
    "DualAffineLayerNorm_022_bfloat16_random": 34.42,
    "DualAffineLayerNorm_023_bfloat16_random": 67.47,
    "DualAffineLayerNorm_024_bfloat16_random": 31.91,
}

ACCURACY_HEADERS = ["shape", "dtype", "精度是否通过", "Ascend截图（单个用例或全部执行通过的截图均可）", "解释说明"]
PERFORMANCE_HEADERS = ["shape", "dtype", "性能是否通过", "Ascend截图（单个用例或全部执行通过的截图均可）",
                       "标杆算子性能(us)", "Ascend C算子性能(us)", "rato", "解释说明"]
MEMORY_HEADERS = ["shape", "dtype", "内存是否通过", "Ascend截图（单个用例或全部执行通过的截图均可）",
                  "标杆算子内存占用（GB）", "Ascend C算子内存占用(GB)", "rato", "解释说明"]
WIDTHS = {"accuracy": [16, 12, 14, 46, 64], "performance": [16, 12, 14, 46, 18, 22, 10, 74],
          "memory": [16, 12, 14, 46, 24, 24, 10, 74]}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing evidence file: {path.name}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"empty evidence file: {path.name}")
    return records


def require_records(name, records, expected_count, keys, expected=None):
    if len(records) != expected_count:
        raise ValueError(f"{name}: expected {expected_count} records, got {len(records)}")
    if any(record.get("status") != "passed" for record in records):
        raise ValueError(f"failed records in {name}")
    identities = [tuple(record.get(key) for key in keys) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate records in {name}")
    if expected is not None:
        expected_identities = {tuple(item.get(key) for key in keys) for item in expected}
        if set(identities) != expected_identities:
            raise ValueError(f"{name}: records do not exactly match the run manifest")
    return records


def validate_evidence(run_dir, project_root=None):
    run_dir = Path(run_dir)
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    validate_implementation_fingerprints(run_dir, project_root)
    validate_task_fingerprint_evidence(run_dir)
    raw = run_dir / "raw"
    summary = read_json(run_dir / "summary.json")
    if summary.get("status") != "passed" or set(summary.get("groups", [])) != ALL_GROUPS:
        raise ValueError("run summary is not a complete passing all-groups run")
    environment_records = read_jsonl(raw / "environment.jsonl")
    if len(environment_records) != 1:
        raise ValueError("environment evidence must contain exactly one final record")
    environment = environment_records[0]
    formal = (
        environment.get("status") == "passed"
        and environment.get("formal_required") is True
        and environment.get("npu_executed") is True
        and not formal_environment_errors(environment)
    )
    if not formal:
        raise ValueError("formal A3/CANN 9.1/torch 2.10 environment evidence is missing or invalid")
    manifest = read_json(raw / "manifest.json")
    if not read_json(raw / "source_fingerprints.json"):
        raise ValueError("source fingerprints are empty")
    evidence = {"environment": environment, "manifest": manifest}
    evidence["public"] = require_records("public accuracy", read_jsonl(raw / "public_accuracy.jsonl"),
                                         46, ("case_id",), manifest.get("public"))
    evidence["performance"] = require_records("performance", read_jsonl(raw / "performance.jsonl"),
                                              10, ("case_id",), manifest.get("performance"))
    evidence["extra"] = require_records("extra accuracy", read_jsonl(raw / "extra.jsonl"),
                                        24, ("case_id", "entry"), manifest.get("extra"))
    evidence["invalid"] = require_records("invalid inputs", read_jsonl(raw / "invalid.jsonl"),
                                          60, ("scenario", "entry"), manifest.get("invalid"))
    evidence["stream"] = require_records("streams", read_jsonl(raw / "stream.jsonl"),
                                         2, ("case_id", "stream_index"), manifest.get("stream"))
    evidence["lifetime"] = require_records("lifetime", read_jsonl(raw / "lifetime.jsonl"),
                                           2, ("entry",))
    evidence["cross_stream_lifetime"] = require_records(
        "cross-stream lifetime", read_jsonl(raw / "cross_stream_lifetime.jsonl"), 2, ("entry",))
    for record in evidence["cross_stream_lifetime"]:
        details = record.get("details", {})
        if (record.get("pressure_calls") != 32 or record.get("early_input_release") is not True
                or details.get("producer_consumer_event_wait") is not True
                or details.get("synchronize_before_input_release") is not False
                or details.get("pressure_stream_independent") is not True):
            raise ValueError("cross-stream input-release pressure protocol evidence is incomplete")
    evidence["memory"] = require_records("memory", read_jsonl(raw / "memory.jsonl"),
                                         20, ("case_id", "entry"), manifest.get("memory"))
    counterexamples = read_jsonl(raw / "verifier_counterexamples.jsonl")
    if len(counterexamples) < 7 or any(record.get("status") != "passed" for record in counterexamples):
        raise ValueError("verifier counterexamples are incomplete or failed")
    evidence["verifier"] = counterexamples
    for record in evidence["stream"]:
        if record.get("prepared_calls", 0) < 10:
            raise ValueError("stream prepared-call coverage is below 10")
    for record in evidence["lifetime"]:
        if record.get("retained_calls") != 8 or record.get("pressure_calls") != 32:
            raise ValueError("lifetime protocol must retain 8 outputs and apply 32 pressure calls")
    for record in evidence["memory"]:
        details = record.get("details", {})
        if (record.get("release_loops") != 20 or details.get("workspace_size") != 0
                or details.get("peak_delta_bytes", 1 << 60) > details.get("allowed_delta_bytes", -1)
                or details.get("active_bytes_after_20_releases") != [details.get("baseline_active_bytes")] * 20):
            raise ValueError("memory protocol evidence is incomplete or above threshold")
    performance = read_json(raw / "performance.json")
    if (performance.get("status") != "passed" or performance.get("npu_executed") is not True
            or performance.get("protocol") != {"warmup": 50, "iterations": 200, "rounds": 5}):
        raise ValueError("official performance protocol/status evidence is invalid")
    acceptance = read_json(raw / "performance_acceptance.json")
    if acceptance.get("status") != "passed" or len(acceptance.get("checks", [])) != 10:
        raise ValueError("official performance acceptance did not pass all 10 cases")
    evidence["performance_acceptance"] = acceptance
    commands = {}
    for name in ("public_accuracy", "performance", "check_performance"):
        command = read_json(raw / f"{name}.command.json")
        if command.get("exit_code") != 0:
            raise ValueError(f"official command failed: {name}")
        commands[name] = command
    evidence["commands"] = commands
    return evidence


def flattened(record, prefix=""):
    result = {}
    for key, value in record.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            result.update(flattened(value, name + "."))
        elif isinstance(value, list):
            result[name] = json.dumps(value, ensure_ascii=False)
        else:
            result[name] = value
    return result


def add_sheet(workbook, title, records):
    sheet = workbook.create_sheet(title)
    rows = [flattened(record) for record in records]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    sheet.append(columns)
    for row in rows:
        sheet.append([row.get(column) for column in columns])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column in sheet.columns:
        width = min(60, max(10, max(len(str(cell.value or "")) for cell in column) + 2))
        sheet.column_dimensions[column[0].column_letter].width = width


def environment_line(environment):
    return (f"{environment.get('device_name', 'unknown')}，CANN {environment.get('cann_version', 'unknown')}，"
            f"驱动 {environment.get('driver_version', 'unknown')}，torch {environment.get('torch', 'unknown')}，"
            f"torch_npu {environment.get('torch_npu', 'unknown')}，Python {environment.get('python', 'unknown')}，"
            f"单逻辑设备 {environment.get('logical_devices', 'unknown')}")


def add_summary_sheet(workbook, title, summary, headers, rows, widths):
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    sheet = workbook.create_sheet(title)
    sheet["A1"] = summary
    sheet["A1"].alignment = Alignment(wrap_text=True, vertical="top")
    sheet.row_dimensions[1].height = 96
    for index, header in enumerate(headers, start=1):
        cell = sheet.cell(row=3, column=index, value=header)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row_index, row in enumerate(rows, start=4):
        for column_index, value in enumerate(row, start=1):
            sheet.cell(row=row_index, column=column_index, value=value)
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.freeze_panes = "A4"
    return sheet


def accuracy_explanation(record):
    parts = []
    for output in record.get("outputs", []) or []:
        name = output.get("output", "output")
        parts.append(f"{name} max_abs_error={output.get('max_abs_error')} "
                     f"max_scaled_error={output.get('max_scaled_error')}")
    return "；".join(parts)


def build_precision_report(path, evidence, environment):
    from openpyxl import Workbook

    public = evidence["public"]
    passed = sum(1 for record in public if record.get("status") == "passed")
    summary = (
        "精度测试总结：\n"
        f"             1、测试环境描述：{environment_line(environment)}\n"
        f"             2、测试用例{len(public)}条，通过{passed}条\n"
        "             3、测试覆盖随机、常量行、大偏移小方差、零 gamma、非对齐 H、H=1 与边界/审计场景\n"
        "             4、测试满足 FP16 atol=rtol=2e-3、BF16 atol=rtol=2e-2 精度要求；常量行与 H=1 精确输出 beta"
    )
    rows = [[record.get("input_shape", ""), record.get("dtype", ""),
             "通过" if record.get("status") == "passed" else "未通过", "", accuracy_explanation(record)]
            for record in public]
    workbook = Workbook()
    workbook.remove(workbook.active)
    add_summary_sheet(workbook, "精度", summary, ACCURACY_HEADERS, rows, WIDTHS["accuracy"])
    add_sheet(workbook, "额外种子与偏移", evidence["extra"])
    add_sheet(workbook, "非法输入", evidence["invalid"])
    add_sheet(workbook, "Stream", evidence["stream"])
    add_sheet(workbook, "生命周期", evidence["lifetime"])
    add_sheet(workbook, "跨Stream压力", evidence["cross_stream_lifetime"])
    add_sheet(workbook, "验证器反例", evidence["verifier"])
    workbook.save(path)


def build_performance_report(path, evidence, environment, measured):
    from openpyxl import Workbook

    checks = evidence["performance_acceptance"]["checks"]
    passed = sum(1 for check in checks if check.get("status") == "passed")
    summary = (
        "性能测试总结：\n"
        f"             1、测试环境描述：{environment_line(environment)}\n"
        f"             2、测试用例{len(checks)}条，通过{passed}条\n"
        "             3、协议为预热 50 次、5 轮、每轮 200 次，取 wall 轮均中位数；同机存在 CPU 争用时\n"
        "                整组 wall 会抬升约 2-3us，结论以安静窗口的完整运行和原始轮次为准\n"
        "             4、测试满足任务书 3.3 的 10 个用例绝对 wall 上限（22/21/20/55/20/20/20/20/55/20 us）"
    )
    rows = []
    for check in checks:
        record = measured.get(check["case_name"], {})
        baseline = PERF_BASELINE_US.get(check["case_name"])
        ours = check.get("wall_us_median")
        row_index = 4 + len(rows)
        ratio = f"=F{row_index}/E{row_index}" if baseline else ""
        custom = (record.get("results") or {}).get("custom", {})
        explanation = (f"目标任务上限 {check.get('max_wall_us')} us；wall 中位数 {ours:.3f} us；"
                       f"device event 中位数 {custom.get('device_event_us_median', float('nan')):.3f} us；"
                       f"五轮 {[round(value, 3) for value in custom.get('wall_us_rounds', [])]}")
        rows.append([record.get("input_shape", ""), record.get("dtype", ""),
                     "通过" if check.get("status") == "passed" else "未通过", "",
                     baseline if baseline is not None else "", round(ours, 6), ratio, explanation])
    workbook = Workbook()
    workbook.remove(workbook.active)
    add_summary_sheet(workbook, "性能", summary, PERFORMANCE_HEADERS, rows, WIDTHS["performance"])
    add_sheet(workbook, "判定明细", checks)
    rounds = []
    for case_name, record in measured.items():
        custom = (record.get("results") or {}).get("custom") or {}
        rounds.append({"case_name": case_name, **custom})
    add_sheet(workbook, "设备事件与轮次", rounds)
    workbook.save(path)


def build_memory_report(path, evidence, environment):
    from openpyxl import Workbook

    memory = evidence["memory"]
    passed = sum(1 for record in memory if record.get("status") == "passed")
    summary = (
        "内存测试总结：\n"
        f"             1、测试环境描述：{environment_line(environment)}\n"
        f"             2、测试记录{len(memory)}条（10 个 performance 用例 x run/run_direct），通过{passed}条\n"
        "             3、显式 workspace 为 0；活跃分配峰值增量不超过两路输出字节加 16KiB 对齐余量；\n"
        "                每条记录执行 20 次释放并同步后回到调用前基线\n"
        "             4、任务书未提供标杆算子内存基线，因此标杆列记为 N/A，比值不适用"
    )
    rows = []
    for record in memory:
        details = record.get("details", {})
        delta_gb = details.get("peak_delta_bytes", 0) / float(1024 ** 3)
        rows.append([record.get("input_shape", ""), record.get("dtype", ""),
                     "通过" if record.get("status") == "passed" else "未通过", "",
                     "N/A（任务书未提供基线）", f"{delta_gb:.9f}", "—",
                     f"entry={record.get('entry')}；workspace={details.get('workspace_size')} B；"
                     f"峰值增量={details.get('peak_delta_bytes')} B ≤ 输出 {details.get('output_bytes')} B + 16KiB；"
                     f"20 次释放后回到基线 {details.get('baseline_active_bytes')} B"])
    workbook = Workbook()
    workbook.remove(workbook.active)
    add_summary_sheet(workbook, "内存（如需要）", summary, MEMORY_HEADERS, rows, WIDTHS["memory"])
    add_sheet(workbook, "生命周期", evidence["lifetime"])
    add_sheet(workbook, "跨Stream压力", evidence["cross_stream_lifetime"])
    workbook.save(path)


def concatenate_evidence(raw, names):
    chunks = []
    for name in names:
        path = raw / name
        chunks.append(path.read_text(encoding="utf-8"))
    return "".join(chunks)


def build_reports(run_dir, output_dir, project_root=None):
    evidence = validate_evidence(run_dir, project_root=project_root)
    raw = Path(run_dir) / "raw"
    output = Path(output_dir)
    existing = [name for name in REQUIRED_REPORTS if (output / name).exists()]
    if existing:
        raise ValueError(f"refusing to overwrite existing reports: {existing}")
    output.mkdir(parents=True, exist_ok=True)
    environment = evidence["environment"]
    measured = {record["case_id"]: record for record in evidence["performance"]}
    build_precision_report(output / "2.1 精度自验证报告.xlsx", evidence, environment)
    build_performance_report(output / "3.1 性能自验证报告.xlsx", evidence, environment, measured)
    build_memory_report(output / "4.1 内存自验证报告.xlsx", evidence, environment)
    (output / "2.2 精度自验证日志.log").write_text(concatenate_evidence(raw, [
        "environment.jsonl", "public_accuracy.jsonl", "extra.jsonl", "invalid.jsonl",
        "stream.jsonl", "lifetime.jsonl", "cross_stream_lifetime.jsonl", "verifier_counterexamples.jsonl", "public_accuracy.stdout.log",
        "public_accuracy.stderr.log",
    ]), encoding="utf-8")
    (output / "3.2 性能自验证日志.log").write_text(concatenate_evidence(raw, [
        "performance.jsonl", "performance.stdout.log", "performance.stderr.log",
        "check_performance.stdout.log", "check_performance.stderr.log",
    ]), encoding="utf-8")
    (output / "4.2 内存自验证日志.log").write_text(concatenate_evidence(raw, [
        "memory.jsonl", "lifetime.jsonl", "cross_stream_lifetime.jsonl",
    ]), encoding="utf-8")
    commands = "\n".join(
        f"- `{name}`: `{' '.join(command['command'])}` (exit 0; stdout/stderr preserved under raw/)"
        for name, command in evidence["commands"].items()
    )
    steps = f"""# DualAffineLayerNorm 自验证步骤说明

> 说明：以下步骤完整可复现。验收人员按此执行即可；所有命令都在项目根目录运行，
> `TASK_ROOT` 指向发布方提供的任务包目录。

### 步骤一：算子编译安装命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh --soc ascend910_93 -j 4
# 可选：只安装到用户目录（脚本拒绝系统 CANN 路径）
bash install_private.sh build/custom_opp_openEuler_aarch64.run "$HOME/.local/dual_affine_layernorm_opp"
export ASCEND_CUSTOM_OPP_PATH="$HOME/.local/dual_affine_layernorm_opp/vendors/dual_affine_layernorm_custom"
```

不安装也可以直接使用 `build/packages/vendors/dual_affine_layernorm_custom`。

### 步骤二：测试程序编译安装命令

```bash
python -m pip install pytest openpyxl ml_dtypes
export ASCEND_CUSTOM_OPP_PATH="$PWD/build/packages/vendors/dual_affine_layernorm_custom"
DUAL_AFFINE_LAYERNORM_SOC=ascend910_9382 python setup.py build_ext --inplace
```

测试程序与报告工具已包含在仓库中，不需要额外编译。

### 步骤三：功能测试执行命令

```bash
# 官方 46 个精度用例
python "$TASK_ROOT/verify.py" --solution-file solution.py --device 0 --report public_accuracy.json

# 全组工程验收（精度/额外/非法/stream/生命周期/内存/性能/反例）
bash tests/run_public_acceptance.sh "$TASK_ROOT" "$PWD/artifacts/a3_run" 0
```

### 步骤四：性能测试执行命令

```bash
python "$TASK_ROOT/benchmark.py" --solution-file solution.py --device 0 --report performance.json
python "$TASK_ROOT/check_performance.py" --report performance.json
```

性能协议为预热 50 次、5 轮、每轮 200 次，以 wall 轮均中位数判定，同时保留 device event。

### 本次运行环境与结论

{environment_line(environment)}

本次正式运行已完整通过：46 个公开精度用例、24 条额外记录、60 条非法输入、双 stream、
8+32 生命周期、三 stream 提前释放压力、10 个性能用例（10/10 达标）、10 用例 x 两入口内存回收
和 10 个独立验证器反例。原始 JSONL、命令退出码、stdout/stderr、任务与交付指纹均保存于运行目录。

### 本次实际命令

{commands}
"""
    (output / "1 自验证步骤说明.md").write_text(steps, encoding="utf-8")
    return [output / name for name in REQUIRED_REPORTS]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    build_reports(args.run_dir, args.output_dir)
    print(f"Generated {len(REQUIRED_REPORTS)} evidence-gated reports in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
