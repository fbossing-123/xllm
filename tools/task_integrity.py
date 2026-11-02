"""Authenticate the supplied verifier, generator, golden and performance contract."""

import hashlib
import json
from pathlib import Path, PurePosixPath


TASK_SHA256SUMS_SHA256 = "a748ed2633ef45eae8cca9e6e2066a7c02fe0c4fb6db006a321ec20a3c6e341e"
SUPPLIED_TASKBOOK_DISK_SHA256 = "eb836858030f4396ded15b2c2e6f401e2cd22eafd1671e78f3b52c363d059756"
OFFICIAL_TASK_FILES = {
    "README.md": "ac8b9b3753864bd7ef0c49811eb81b41fb8741a8d46dee722e8e8549141525c6",
    "benchmark.py": "84a198138a226f6185348c81045dde9bfac1c70e98b0d8e3a529c34261db868d",
    "case.json": "b3ec003226f191a40c9be6c34d5a14ddfbbe1db9808449d351544dd74d624365",
    "check_performance.py": "52f3fbb1280284c74585fbe2dfafeef91d8d77ee3ef1a8bbe4495c0c9bc0f2c5",
    "coverage_requirements.json": "b9bb05590ed2e5d827762d590caa4c2a38ebfab47ead05277b5b3e7471bdfaf5",
    "generate_data.py": "39c73731b0afebd242b2fc8e3e85df24c4b9c5c423ebe9adc1e4140507b2292d",
    "golden.py": "a43f85e03d4ebcc9ae32f4564e6f7a76ea139f091ac259974ab92d9d3c64b151",
    "op.json": "1bdfae7517f5ea1c614d4cc9f39bfa832792562b5c0402de295d807fcabce99b",
    "performance_targets.json": "56585e27b036b786a20330c666f5cda6852eb223fe54c3cabf54afbc37793371",
    "verify.py": "94926d5505b9252ff3b1be406a525f6fe898cba84d522a4a8b18c9b2f1ea59a9",
    "题目1-双分支LayerNorm算子任务书.md": "eb836858030f4396ded15b2c2e6f401e2cd22eafd1671e78f3b52c363d059756",
}


def authenticate_task_root(task_root, expected_checksum_manifest_sha256=TASK_SHA256SUMS_SHA256):
    """Verify the pinned checksum manifest before any supplied Python is imported."""
    root = Path(task_root).resolve()
    checksum_path = root / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise ValueError("official task SHA256SUMS.txt is missing")
    checksum_bytes = checksum_path.read_bytes()
    checksum_sha256 = hashlib.sha256(checksum_bytes).hexdigest()
    if checksum_sha256 != expected_checksum_manifest_sha256:
        raise ValueError("official task SHA256SUMS.txt fingerprint mismatch")
    entries = {}
    for line in checksum_bytes.decode("utf-8").splitlines():
        fields = line.split("  ")
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError("official task SHA256SUMS.txt has an invalid line")
        digest, name = fields
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or name in entries:
            raise ValueError(f"official task checksum path is invalid: {name}")
        entries[name] = digest
    if set(entries) != set(OFFICIAL_TASK_FILES):
        raise ValueError("official task checksum file set is incomplete or unexpected")
    files = {}
    known_discrepancies = []
    for name, expected in entries.items():
        path = root / Path(*PurePosixPath(name).parts)
        if name == "题目1-双分支LayerNorm算子任务书.md" and not path.is_file():
            # The supplied ZIP omits this prefix on disk although SHA256SUMS.txt includes it.
            path = root / "双分支LayerNorm算子任务书.md"
        if not path.is_file():
            raise ValueError(f"official task file is missing: {name}")
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            supplied_taskbook = (
                name == "题目1-双分支LayerNorm算子任务书.md"
                and checksum_sha256 == TASK_SHA256SUMS_SHA256
                and actual == SUPPLIED_TASKBOOK_DISK_SHA256
            )
            if not supplied_taskbook:
                raise ValueError(f"official task file fingerprint mismatch: {name}")
            known_discrepancies.append(name)
        files[name] = {"sha256": actual, "bytes": len(content)}
    return {"checksum_manifest_sha256": checksum_sha256, "files": files,
            "known_discrepancies": known_discrepancies}


def validate_task_fingerprint_evidence(run_dir):
    path = Path(run_dir) / "raw" / "task_fingerprints.json"
    if not path.is_file():
        raise ValueError("official task fingerprint evidence is missing")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    actual = {name: metadata.get("sha256") for name, metadata in evidence.get("files", {}).items()}
    if (evidence.get("checksum_manifest_sha256") != TASK_SHA256SUMS_SHA256
            or actual != OFFICIAL_TASK_FILES
            or evidence.get("known_discrepancies") != ["题目1-双分支LayerNorm算子任务书.md"]):
        raise ValueError("official task fingerprint evidence is incomplete or does not match the supplied package")
    return evidence
