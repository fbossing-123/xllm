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


def save_workbook(path, sheets):
    from openpyxl import Workbook
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, records in sheets:
        add_sheet(workbook, title, records)
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
    save_workbook(output / "2.1 精度自验证报告.xlsx", [
        ("Public46", evidence["public"]), ("Extra24", evidence["extra"]),
        ("Invalid60", evidence["invalid"]), ("Streams", evidence["stream"]),
        ("Lifetime", evidence["lifetime"]), ("CrossStreamLifetime", evidence["cross_stream_lifetime"]),
        ("VerifierCounterexamples", evidence["verifier"]),
    ])
    performance_rows = []
    measured = {record["case_id"]: record for record in evidence["performance"]}
    for check in evidence["performance_acceptance"]["checks"]:
        row = dict(check)
        custom = measured[check["case_name"]]["results"]["custom"]
        row["device_event_us_median"] = custom["device_event_us_median"]
        row["wall_us_rounds"] = custom["wall_us_rounds"]
        row["device_event_us_rounds"] = custom["device_event_us_rounds"]
        performance_rows.append(row)
    save_workbook(output / "3.1 性能自验证报告.xlsx", [("Performance10", performance_rows)])
    save_workbook(output / "4.1 内存自验证报告.xlsx", [
        ("Memory", evidence["memory"]), ("Lifetime", evidence["lifetime"]),
        ("CrossStreamLifetime", evidence["cross_stream_lifetime"]),
    ])
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
    environment = evidence["environment"]
    commands = "\n".join(
        f"- `{name}`: `{' '.join(command['command'])}` (exit 0; stdout/stderr preserved under raw/)"
        for name, command in evidence["commands"].items()
    )
    steps = f"""# DualAffineLayerNorm 自验证步骤说明

## 验证结论

本报告由证据门禁从一次完整、全部通过的正式设备运行生成。环境：{environment['device_name']}，
CANN {environment['cann_version']}，torch {environment['torch']}，torch_npu {environment['torch_npu']}。
证据覆盖 46 个公开精度用例、24 条额外记录、60 条非法输入记录、双 stream、8+32 生命周期、
三 stream 提前释放压力、10 个性能用例、10 用例×两入口内存回收和至少 7 个独立验证器反例。

## 复现

1. 在 aarch64 Linux 主机安装 CANN 9.1.0，并安装与目标 Python 匹配的 torch/torch_npu 2.10。
2. `bash build.sh --soc ascend910_93 -j 4`，随后用 `install_private.sh` 仅安装到用户目录。
3. `DUAL_AFFINE_LAYERNORM_SOC=ascend910_9382 python setup.py build_ext --inplace`。
4. 运行 `bash tests/run_public_acceptance.sh TASK_ROOT OUTPUT_DIR 0`。
5. 只有 `OUTPUT_DIR/summary.json` 为 complete passing 时，运行
   `python tools/build_reports.py --run-dir OUTPUT_DIR --output-dir task_submission`。

## 本次官方命令

{commands}

性能协议为预热 50 次、5 轮、每轮 200 次，以 wall 轮均中位数判定，并保留 device event。
内存协议要求 10 个 performance 用例分别验证两入口，workspace=0、峰值不超过两路输出加 16KiB，
每条 20 次释放后回到基线。三 stream 协议在计算完成前释放输入并施加另一 stream 的 32 次写入压力。
所有原始 JSONL、命令退出码、stdout/stderr、官方任务及当前交付源码/二进制指纹均保存在运行目录。
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
