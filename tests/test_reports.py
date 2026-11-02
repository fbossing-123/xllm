import json
from pathlib import Path

import pytest

from tools.build_reports import REQUIRED_REPORTS, build_reports, validate_evidence
from tools.fingerprint import create_manifest
from tools.package_submission import package_submission
from tools.task_integrity import OFFICIAL_TASK_FILES, TASK_SHA256SUMS_SHA256


GROUPS = ["accuracy", "extra", "invalid", "lifetime", "memory", "performance", "stream", "verifier"]


def jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def synthetic_evidence(tmp_path):
    """Synthetic gate fixture only; never a real device acceptance result."""
    run = tmp_path / "synthetic_fixture"
    raw = run / "raw"
    (run / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text(json.dumps({"status": "passed", "groups": GROUPS}), encoding="utf-8")
    jsonl(raw / "environment.jsonl", [{
        "status": "passed", "formal_required": True, "npu_executed": True,
        "device_name": "Ascend910_9382", "cann_version": "9.1.0", "device_index": 0,
        "python": "3.12.13", "torch": "2.10.0+cpu", "torch_npu": "2.10.0.post4+git5dd8ef3",
        "driver_version": "25.5.t6.b001", "architecture": "aarch64",
        "os_name": "openEuler 24.03 LTS-SP3", "logical_devices": 1,
        "vector_cores": 48, "cube_cores": 24, "platform": "Linux-aarch64",
    }])
    jsonl(raw / "public_accuracy.jsonl", [
        {"case_id": f"case_{i:03d}", "case_name": f"case_{i:03d}", "status": "passed", "outputs": []}
        for i in range(46)
    ])
    jsonl(raw / "performance.jsonl", [
        {"case_id": f"perf_{i}", "status": "passed", "results": {"custom": {
            "wall_us_median": 10 + i, "device_event_us_median": 8 + i,
            "wall_us_rounds": [10 + i] * 5, "device_event_us_rounds": [8 + i] * 5,
        }}} for i in range(10)
    ])
    (raw / "performance.json").write_text(json.dumps({
        "status": "passed", "npu_executed": True,
        "protocol": {"warmup": 50, "iterations": 200, "rounds": 5},
    }), encoding="utf-8")
    (raw / "performance_acceptance.json").write_text(json.dumps({
        "status": "passed", "checks": [
            {"case_name": f"perf_{i}", "status": "passed", "wall_us_median": 10 + i, "max_wall_us": 55}
            for i in range(10)
        ],
    }), encoding="utf-8")
    jsonl(raw / "extra.jsonl", [
        {"case_id": f"extra_{i // 2}", "entry": "run" if i % 2 == 0 else "run_direct", "status": "passed"}
        for i in range(24)
    ])
    jsonl(raw / "invalid.jsonl", [
        {"scenario": f"invalid_{i // 3}", "entry": ("prepare", "run", "run_direct")[i % 3], "status": "passed"}
        for i in range(60)
    ])
    jsonl(raw / "stream.jsonl", [
        {"case_id": "case_005", "stream_index": 0, "prepared_calls": 10, "status": "passed"},
        {"case_id": "case_024", "stream_index": 1, "prepared_calls": 10, "status": "passed"},
    ])
    jsonl(raw / "lifetime.jsonl", [
        {"entry": entry, "retained_calls": 8, "pressure_calls": 32, "status": "passed"}
        for entry in ("run", "run_direct")
    ])
    jsonl(raw / "cross_stream_lifetime.jsonl", [
        {"case_id": "case_005", "entry": entry, "pressure_calls": 32,
         "early_input_release": True, "status": "passed", "details": {
             "producer_consumer_event_wait": True, "synchronize_before_input_release": False,
             "pressure_stream_independent": True,
         }} for entry in ("run", "run_direct")
    ])
    jsonl(raw / "memory.jsonl", [
        {"case_id": f"perf_{i}", "entry": entry, "release_loops": 20, "status": "passed", "details": {
            "workspace_size": 0, "peak_delta_bytes": 128, "output_bytes": 128,
            "allowed_delta_bytes": 128 + 16384, "baseline_active_bytes": 1000,
            "active_bytes_after_20_releases": [1000] * 20,
        }} for i in range(10) for entry in ("run", "run_direct")
    ])
    jsonl(raw / "verifier_counterexamples.jsonl", [
        {"name": f"counterexample_{i}", "status": "passed"} for i in range(10)
    ])
    manifest = {
        "public": [{"case_id": f"case_{i:03d}", "input_sha256": f"hash_{i}"} for i in range(46)],
        "performance": [{"case_id": f"perf_{i}"} for i in range(10)],
        "memory": [{"case_id": f"perf_{i}", "entry": entry}
                   for i in range(10) for entry in ("run", "run_direct")],
        "extra": [{"case_id": f"extra_{i}", "entry": entry} for i in range(12) for entry in ("run", "run_direct")],
        "invalid": [{"scenario": f"invalid_{i}", "entry": entry} for i in range(20) for entry in ("prepare", "run", "run_direct")],
        "stream": [{"case_id": "case_005", "stream_index": 0}, {"case_id": "case_024", "stream_index": 1}],
    }
    (raw / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (raw / "source_fingerprints.json").write_text(json.dumps({"solution.py": "fixture-hash"}), encoding="utf-8")
    (raw / "task_fingerprints.json").write_text(json.dumps({
        "checksum_manifest_sha256": TASK_SHA256SUMS_SHA256,
        "files": {name: {"sha256": digest, "bytes": 1} for name, digest in OFFICIAL_TASK_FILES.items()},
        "known_discrepancies": ["题目1-双分支LayerNorm算子任务书.md"],
    }), encoding="utf-8")
    project = tmp_path / "implementation"
    package = project / "dual_affine_layernorm"
    package.mkdir(parents=True, exist_ok=True)
    (project / "solution.py").write_text("synthetic tested source\n", encoding="utf-8")
    loaded = []
    for name in ("_dual_affine_layer_norm.cpython-312-aarch64-linux-gnu.so",
                 "libdual_affine_layer_norm_torch_kernel.so", "libcust_opapi.so"):
        path = package / name
        path.write_bytes(("synthetic binary: " + name).encode())
        loaded.append({"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()})
    implementation = create_manifest(project)
    implementation["loaded_binaries"] = loaded
    (raw / "implementation_fingerprints.json").write_text(json.dumps(implementation), encoding="utf-8")
    for name in ("public_accuracy", "performance", "check_performance"):
        (raw / f"{name}.command.json").write_text(json.dumps({
            "command": ["python", f"{name}.py"], "exit_code": 0,
            "stdout_log": f"{name}.stdout.log", "stderr_log": f"{name}.stderr.log",
        }), encoding="utf-8")
        (raw / f"{name}.stdout.log").write_text("synthetic fixture\n", encoding="utf-8")
        (raw / f"{name}.stderr.log").write_text("", encoding="utf-8")
    return run


def fixture_project(run):
    return run.parent / "implementation"


def test_complete_synthetic_fixture_generates_seven_reports_only_in_tmp(tmp_path):
    run = synthetic_evidence(tmp_path)
    output = tmp_path / "reports"
    build_reports(run, output, project_root=fixture_project(run))
    assert {path.name for path in output.iterdir()} == set(REQUIRED_REPORTS)
    for name in REQUIRED_REPORTS:
        assert (output / name).stat().st_size > 0


def test_gate_rejects_failed_wrong_environment_and_incomplete_counts(tmp_path):
    run = synthetic_evidence(tmp_path)
    environment = (run / "raw/environment.jsonl").read_text(encoding="utf-8")
    (run / "raw/environment.jsonl").write_text(environment.replace("Ascend910_9382", "Ascend910B4"), encoding="utf-8")
    with pytest.raises(ValueError, match="formal"):
        validate_evidence(run, project_root=fixture_project(run))

    run = synthetic_evidence(tmp_path / "failed")
    records = [json.loads(line) for line in (run / "raw/extra.jsonl").read_text().splitlines()]
    records[0]["status"] = "failed"
    jsonl(run / "raw/extra.jsonl", records)
    with pytest.raises(ValueError, match="failed"):
        validate_evidence(run, project_root=fixture_project(run))

    run = synthetic_evidence(tmp_path / "incomplete")
    lines = (run / "raw/invalid.jsonl").read_text().splitlines()
    (run / "raw/invalid.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="60"):
        validate_evidence(run, project_root=fixture_project(run))


def test_cross_stream_pressure_evidence_is_mandatory(tmp_path):
    run = synthetic_evidence(tmp_path)
    (run / "raw/cross_stream_lifetime.jsonl").unlink()
    with pytest.raises(ValueError, match="cross.stream"):
        validate_evidence(run, project_root=fixture_project(run))


def test_memory_gate_requires_all_ten_performance_cases(tmp_path):
    run = synthetic_evidence(tmp_path)
    lines = (run / "raw/memory.jsonl").read_text(encoding="utf-8").splitlines()
    (run / "raw/memory.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="memory.*20"):
        validate_evidence(run, project_root=fixture_project(run))


def test_report_gate_rejects_non_exact_formal_python(tmp_path):
    run = synthetic_evidence(tmp_path)
    path = run / "raw/environment.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("3.12.13", "3.12.12"), encoding="utf-8")
    with pytest.raises(ValueError, match="formal"):
        validate_evidence(run, project_root=fixture_project(run))


def test_reports_refuse_to_overwrite_existing_reports(tmp_path):
    run = synthetic_evidence(tmp_path)
    output = tmp_path / "reports"
    output.mkdir()
    (output / "2.1 精度自验证报告.xlsx").write_bytes(b"existing-passing-report")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_reports(run, output, project_root=fixture_project(run))
    assert (output / "2.1 精度自验证报告.xlsx").read_bytes() == b"existing-passing-report"


def test_development_package_cannot_be_named_as_passing_submission(tmp_path):
    with pytest.raises(ValueError, match="task_submission.zip"):
        package_submission(tmp_path, tmp_path / "task_submission.zip", development=True)


def test_formal_package_rejects_source_changed_after_acceptance(tmp_path):
    project = tmp_path / "project"
    package = project / "dual_affine_layernorm"
    package.mkdir(parents=True)
    (project / "solution.py").write_text("tested = True\n", encoding="utf-8")
    loaded = []
    for name in ("_dual_affine_layer_norm.cpython-312-aarch64-linux-gnu.so",
                 "libdual_affine_layer_norm_torch_kernel.so", "libcust_opapi.so"):
        path = package / name
        path.write_bytes(("tested binary: " + name).encode())
        loaded.append({"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()})
    run = synthetic_evidence(tmp_path / "evidence")
    implementation = create_manifest(project)
    implementation["loaded_binaries"] = loaded
    (run / "raw/implementation_fingerprints.json").write_text(json.dumps(implementation), encoding="utf-8")
    submission = project / "task_submission"
    submission.mkdir()
    for name in REQUIRED_REPORTS:
        (submission / name).write_bytes(b"synthetic report")

    valid = package_submission(project, tmp_path / "valid/task_submission.zip", run)
    assert valid["status"] == "formal_A3_passed"

    (project / "solution.py").write_text("tested = False  # changed after acceptance\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation.*changed|implementation.*mismatch"):
        package_submission(project, tmp_path / "changed/task_submission.zip", run)


def test_report_generation_rejects_source_changed_after_acceptance(tmp_path):
    project = tmp_path / "project"
    package = project / "dual_affine_layernorm"
    package.mkdir(parents=True)
    (project / "solution.py").write_text("tested = True\n", encoding="utf-8")
    loaded = []
    for name in ("_dual_affine_layer_norm.cpython-312-aarch64-linux-gnu.so",
                 "libdual_affine_layer_norm_torch_kernel.so", "libcust_opapi.so"):
        path = package / name
        path.write_bytes(name.encode())
        loaded.append({"path": str(path), "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest()})
    run = synthetic_evidence(tmp_path / "evidence")
    implementation = create_manifest(project)
    implementation["loaded_binaries"] = loaded
    (run / "raw/implementation_fingerprints.json").write_text(json.dumps(implementation), encoding="utf-8")
    (project / "solution.py").write_text("changed after acceptance\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation.*changed|implementation.*mismatch"):
        build_reports(run, tmp_path / "reports", project_root=project)
