import hashlib
import json
from pathlib import Path

import pytest


TASK_NAMES = [
    "README.md", "benchmark.py", "case.json", "check_performance.py",
    "coverage_requirements.json", "generate_data.py", "golden.py", "op.json",
    "performance_targets.json", "verify.py", "题目1-双分支LayerNorm算子任务书.md",
]


def synthetic_task(root):
    root.mkdir(parents=True)
    lines = []
    for name in TASK_NAMES:
        disk_name = "双分支LayerNorm算子任务书.md" if name.startswith("题目1-") else name
        path = root / disk_name
        path.write_text("official synthetic fixture: " + name + "\n", encoding="utf-8")
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
    checksums = root / "SHA256SUMS.txt"
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksums


def rewrite_checksums(root):
    lines = [
        f"{hashlib.sha256((root / ('双分支LayerNorm算子任务书.md' if name.startswith('题目1-') else name)).read_bytes()).hexdigest()}  {name}"
        for name in TASK_NAMES
    ]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_task_authentication_rejects_changed_performance_targets(tmp_path):
    from tools.task_integrity import authenticate_task_root

    root = tmp_path / "task"
    checksums = synthetic_task(root)
    pinned = hashlib.sha256(checksums.read_bytes()).hexdigest()
    evidence = authenticate_task_root(root, expected_checksum_manifest_sha256=pinned)
    assert evidence["files"]["golden.py"]["sha256"]

    (root / "performance_targets.json").write_text('{"max_wall_us": 99999}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="performance_targets.json"):
        authenticate_task_root(root, expected_checksum_manifest_sha256=pinned)

    rewrite_checksums(root)
    with pytest.raises(ValueError, match="SHA256SUMS"):
        authenticate_task_root(root, expected_checksum_manifest_sha256=pinned)


def test_only_the_exact_supplied_taskbook_discrepancy_is_accepted(tmp_path, monkeypatch):
    import tools.task_integrity as integrity

    root = tmp_path / "task"
    checksums = synthetic_task(root)
    name = "题目1-双分支LayerNorm算子任务书.md"
    lines = checksums.read_text(encoding="utf-8").splitlines()
    lines = [("0" * 64 + "  " + name) if line.endswith("  " + name) else line for line in lines]
    checksums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pinned = hashlib.sha256(checksums.read_bytes()).hexdigest()
    taskbook = root / "双分支LayerNorm算子任务书.md"
    actual = hashlib.sha256(taskbook.read_bytes()).hexdigest()
    monkeypatch.setattr(integrity, "TASK_SHA256SUMS_SHA256", pinned)
    monkeypatch.setattr(integrity, "SUPPLIED_TASKBOOK_DISK_SHA256", actual, raising=False)
    evidence = integrity.authenticate_task_root(root, expected_checksum_manifest_sha256=pinned)
    assert evidence["files"][name]["sha256"] == actual
    assert evidence["known_discrepancies"] == [name]

    taskbook.write_text("unverified taskbook contents\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        integrity.authenticate_task_root(root, expected_checksum_manifest_sha256=pinned)


def test_report_gate_rejects_unauthenticated_official_task_evidence(tmp_path):
    from tests.test_reports import fixture_project, synthetic_evidence
    from tools.build_reports import validate_evidence

    run = synthetic_evidence(tmp_path)
    (run / "raw/task_fingerprints.json").write_text(json.dumps({
        "checksum_manifest_sha256": "a748ed2633ef45eae8cca9e6e2066a7c02fe0c4fb6db006a321ec20a3c6e341e",
        "files": {"golden.py": {"sha256": "altered"}},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="task.*fingerprint|official task"):
        validate_evidence(run, project_root=fixture_project(run))
