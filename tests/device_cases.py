"""Frozen acceptance coverage expansion; no device or framework dependency."""

import json
from pathlib import Path


PERFORMANCE_PROTOCOL = {"warmup": 50, "iterations": 200, "rounds": 5}
MIN_PREPARED_CALLS_PER_STREAM = 10
MEMORY_RELEASE_LOOPS = 20
LIFETIME_RETAINED_CALLS = 8
LIFETIME_PRESSURE_CALLS = 32
MIN_VERIFIER_COUNTEREXAMPLES = 7


def load_coverage(task_root):
    path = Path(task_root).resolve() / "coverage_requirements.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(coverage):
    required = (
        "public_cases", "performance_case_ids", "extra_input_case_ids", "extra_entries",
        "negative_scenarios", "negative_entries", "stream_case_ids",
        "min_prepared_calls_per_stream",
    )
    for field in required:
        if field not in coverage:
            raise ValueError(f"official coverage is missing {field}")
    public_cases = coverage["public_cases"]
    if not isinstance(public_cases, dict):
        raise ValueError("public_cases must map case IDs to input hashes")
    public = [
        {"case_id": case_id, "input_sha256": digest}
        for case_id, digest in public_cases.items()
    ]
    known = set(public_cases)
    for field in ("performance_case_ids", "stream_case_ids"):
        unknown = set(coverage[field]) - known
        if unknown:
            raise ValueError(f"{field} contains unknown public cases: {sorted(unknown)}")
    prepared_calls = coverage["min_prepared_calls_per_stream"]
    if type(prepared_calls) is not int or prepared_calls < MIN_PREPARED_CALLS_PER_STREAM:
        raise ValueError("official stream coverage requires at least 10 prepared calls")
    return {
        "public": public,
        "performance": [{"case_id": case_id} for case_id in coverage["performance_case_ids"]],
        "memory": [
            {"case_id": case_id, "entry": entry}
            for case_id in coverage["performance_case_ids"]
            for entry in ("run", "run_direct")
        ],
        "extra": [
            {"case_id": case_id, "entry": entry}
            for case_id in coverage["extra_input_case_ids"]
            for entry in coverage["extra_entries"]
        ],
        "invalid": [
            {"scenario": scenario, "entry": entry}
            for scenario in coverage["negative_scenarios"]
            for entry in coverage["negative_entries"]
        ],
        "stream": [
            {"case_id": case_id, "stream_index": index, "prepared_calls": prepared_calls}
            for index, case_id in enumerate(coverage["stream_case_ids"])
        ],
    }
