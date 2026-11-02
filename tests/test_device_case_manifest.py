import json
from pathlib import Path

import pytest

from tests.device_cases import (
    LIFETIME_PRESSURE_CALLS,
    LIFETIME_RETAINED_CALLS,
    MEMORY_RELEASE_LOOPS,
    MIN_PREPARED_CALLS_PER_STREAM,
    PERFORMANCE_PROTOCOL,
    build_manifest,
)


def coverage_fixture():
    public = [f"case_{index:03d}" for index in range(46)]
    return {
        "public_cases": {name: f"sha-{index}" for index, name in enumerate(public)},
        "performance_case_ids": public[5:15],
        "extra_input_case_ids": [f"extra_{index}" for index in range(12)],
        "extra_entries": ["run", "run_direct"],
        "negative_scenarios": [f"invalid_{index}" for index in range(20)],
        "negative_entries": ["prepare", "run", "run_direct"],
        "stream_case_ids": [public[5], public[24]],
        "min_prepared_calls_per_stream": 10,
    }


def test_manifest_expands_every_required_device_record():
    manifest = build_manifest(coverage_fixture())
    assert len(manifest["public"]) == 46
    assert len(manifest["performance"]) == 10
    assert manifest["memory"] == [
        {"case_id": case_id, "entry": entry}
        for case_id in coverage_fixture()["performance_case_ids"] for entry in ("run", "run_direct")
    ]
    assert len(manifest["extra"]) == 24
    assert {(r["case_id"], r["entry"]) for r in manifest["extra"]} == {
        (f"extra_{index}", entry)
        for index in range(12) for entry in ("run", "run_direct")
    }
    assert len(manifest["invalid"]) == 60
    assert {(r["scenario"], r["entry"]) for r in manifest["invalid"]} == {
        (f"invalid_{index}", entry)
        for index in range(20) for entry in ("prepare", "run", "run_direct")
    }
    assert manifest["stream"] == [
        {"case_id": "case_005", "stream_index": 0, "prepared_calls": 10},
        {"case_id": "case_024", "stream_index": 1, "prepared_calls": 10},
    ]


def test_protocol_counts_are_frozen():
    assert PERFORMANCE_PROTOCOL == {"warmup": 50, "iterations": 200, "rounds": 5}
    assert MIN_PREPARED_CALLS_PER_STREAM == 10
    assert MEMORY_RELEASE_LOOPS == 20
    assert LIFETIME_RETAINED_CALLS == 8
    assert LIFETIME_PRESSURE_CALLS == 32


@pytest.mark.parametrize("field", ["public_cases", "extra_entries", "negative_scenarios"])
def test_manifest_rejects_missing_official_coverage(field):
    coverage = coverage_fixture()
    del coverage[field]
    with pytest.raises(ValueError, match=field):
        build_manifest(coverage)


def test_manifest_rejects_public_case_hash_mismatch():
    coverage = coverage_fixture()
    coverage["public_cases"]["case_000"] = "not-the-official-hash"
    manifest = build_manifest(coverage)
    assert manifest["public"][0]["input_sha256"] == "not-the-official-hash"
    assert len({r["input_sha256"] for r in manifest["public"]}) == 46


def test_evidence_gate_rejects_missing_failed_and_duplicate_records(tmp_path):
    from tests.run_device_acceptance import require_complete_records

    expected = [{"case_id": "a", "entry": "run"}, {"case_id": "b", "entry": "run"}]
    keys = ("case_id", "entry")
    with pytest.raises(ValueError, match="missing"):
        require_complete_records([{"case_id": "a", "entry": "run", "status": "passed"}], expected, keys)
    with pytest.raises(ValueError, match="failed"):
        require_complete_records([
            {"case_id": "a", "entry": "run", "status": "passed"},
            {"case_id": "b", "entry": "run", "status": "failed"},
        ], expected, keys)
    with pytest.raises(ValueError, match="duplicate"):
        require_complete_records([
            {"case_id": "a", "entry": "run", "status": "passed"},
            {"case_id": "a", "entry": "run", "status": "passed"},
            {"case_id": "b", "entry": "run", "status": "passed"},
        ], expected, keys)


def test_evidence_gate_requires_formal_environment_when_requested():
    from tests.run_device_acceptance import validate_environment

    compatibility = {
        "device_name": "Ascend910B4", "cann_version": "9.0.0",
        "torch_npu": "2.5.1", "npu_executed": True,
    }
    assert validate_environment(compatibility, require_formal=False)["status"] == "passed"
    with pytest.raises(ValueError, match="formal"):
        validate_environment(compatibility, require_formal=True)


def formal_environment_fixture():
    return {
        "npu_executed": True, "device_name": "Ascend910_9382", "device_index": 0,
        "cann_installation_info": {"content": "version=9.1.0\n"},
        "python": "3.12.13", "torch": "2.10.0+cpu",
        "torch_npu": "2.10.0.post4+git5dd8ef3", "driver_version": "25.5.t6.b001",
        "architecture": "aarch64", "os_name": "openEuler 24.03 LTS-SP3",
        "platform": "Linux-aarch64", "logical_devices": 1,
        "vector_cores": 48, "cube_cores": 24,
    }


def test_formal_environment_requires_every_exact_target_field():
    from tests.run_device_acceptance import validate_environment

    valid = formal_environment_fixture()
    assert validate_environment(valid, require_formal=True)["status"] == "passed"
    mutations = {
        "python": "3.12.12", "torch": "2.10.1+cpu",
        "torch_npu": "2.10.0.post3+git5dd8ef3", "driver_version": "unknown",
        "os_name": "Ubuntu 24.04", "architecture": "x86_64", "logical_devices": 2,
        "vector_cores": 47, "cube_cores": 23, "device_index": 1,
    }
    for field, value in mutations.items():
        environment = dict(valid)
        environment[field] = value
        with pytest.raises(ValueError, match="formal"):
            validate_environment(environment, require_formal=True)


def test_formal_environment_accepts_newer_backward_compatible_driver():
    from tests.run_device_acceptance import validate_environment

    valid = formal_environment_fixture()
    for newer in ("26.1.1", "25.5.t6.b002", "26.0.rc1"):
        environment = dict(valid)
        environment["driver_version"] = newer
        assert validate_environment(environment, require_formal=True)["status"] == "passed"
    for older in ("25.4.0", "25.5.t5.b999", "24.1.0", "unknown"):
        environment = dict(valid)
        environment["driver_version"] = older
        with pytest.raises(ValueError, match="formal"):
            validate_environment(environment, require_formal=True)


def test_formal_environment_accepts_pinned_torch_npu_commit_without_local_label():
    from tests.run_device_acceptance import validate_environment

    environment = formal_environment_fixture()
    environment["torch_npu"] = "2.10.0.post4"
    environment["torch_npu_git_version"] = "5dd8ef3f9b375b5ae4a83538d5785754148c3302"
    assert validate_environment(environment, require_formal=True)["status"] == "passed"
    for wrong in ({"torch_npu": "2.10.0.post3", "torch_npu_git_version": "5dd8ef3f"},
                  {"torch_npu": "2.10.0.post4", "torch_npu_git_version": "0000000"},
                  {"torch_npu": "2.10.0.post4", "torch_npu_git_version": ""}):
        broken = dict(environment)
        broken.update(wrong)
        with pytest.raises(ValueError, match="formal"):
            validate_environment(broken, require_formal=True)


def test_core_count_query_uses_cann_vector_and_cube_attributes(monkeypatch):
    import ctypes
    from tests import run_device_acceptance as runner

    calls = []
    class Library:
        def aclrtGetDeviceInfo(self, device, attribute, value):
            calls.append((device, attribute))
            value._obj.value = 48 if attribute == 201 else 24
            return 0
    monkeypatch.setattr(ctypes, "CDLL", lambda name: Library())
    assert runner.query_core_counts(0) == (48, 24)
    assert calls == [(0, 201), (0, 102)]


def test_verifier_counterexample_catalog_has_all_independent_failure_modes():
    from tests.verifier_counterexamples import COUNTEREXAMPLE_NAMES

    assert len(COUNTEREXAMPLE_NAMES) >= 7
    assert len(set(COUNTEREXAMPLE_NAMES)) == len(COUNTEREXAMPLE_NAMES)
    for name in ["cpu_output", "input_storage_reuse", "shared_output_storage",
                 "wrong_count", "wrong_shape", "wrong_dtype", "noncontiguous", "input_mutation"]:
        assert name in COUNTEREXAMPLE_NAMES


def test_public_acceptance_wrapper_checks_device_and_runs_fail_closed(project_root):
    script = (project_root / "tests/run_public_acceptance.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    assert "npu-smi info" in script
    assert "--require-formal" in script
    assert "run_device_acceptance.py" in script
    assert "pkill" not in script


def test_invalid_fixture_uses_shape_where_strided_view_is_noncontiguous():
    from tests.run_device_acceptance import select_invalid_base_case

    cases = [
        {"case_name": "scalar", "input_desc": [{"shape": [1, 1]}]},
        {"case_name": "matrix", "input_desc": [{"shape": [2, 8]}]},
    ]
    assert select_invalid_base_case(cases)["case_name"] == "matrix"
    with pytest.raises(ValueError, match="noncontiguous"):
        select_invalid_base_case(cases[:1])


def test_official_accuracy_group_does_not_force_performance_benchmark(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests import run_device_acceptance as runner

    commands = []
    raw = tmp_path / "raw"
    raw.mkdir()

    def fake_run(command, raw_dir, name, cwd, env=None):
        commands.append(name)
        (raw_dir / "public_accuracy.json").write_text(json.dumps({"cases": []}), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner, "run_logged", fake_run)
    runner.run_official(tmp_path, tmp_path, tmp_path / "solution.py", 0, raw,
                        run_accuracy=True, run_performance=False)
    assert commands == ["public_accuracy"]


def test_environment_collection_uses_official_cann_installation_metadata():
    from types import SimpleNamespace
    from tests.run_device_acceptance import collect_environment

    benchmark = SimpleNamespace(environment_info=lambda torch, torch_npu, device: {
        "device_name": "Ascend910_9382",
        "cann_installation_info": {"content": "version=9.1.0\n"},
        "torch": "2.10.0", "torch_npu": "2.10.0", "platform": "Linux-aarch64",
    })
    environment = collect_environment(benchmark, object(), object(), 0)
    assert environment["npu_executed"] is True
    assert environment["cann_installation_info"]["content"] == "version=9.1.0\n"


def test_loaded_binary_fingerprints_cover_actual_relevant_mappings(tmp_path):
    import hashlib
    from tests.run_device_acceptance import loaded_binary_fingerprints

    extension = tmp_path / "_dual_affine_layer_norm.cpython-312-aarch64-linux-gnu.so"
    kernel = tmp_path / "libdual_affine_layer_norm_torch_kernel.so"
    unrelated = tmp_path / "libunrelated.so"
    for path in (extension, kernel, unrelated):
        path.write_bytes(path.name.encode())
    maps = tmp_path / "maps"
    maps.write_text(
        f"00-01 r-xp 00000000 00:00 0 {extension}\n"
        f"01-02 r-xp 00000000 00:00 0 {extension}\n"
        f"02-03 r-xp 00000000 00:00 0 {kernel}\n"
        f"03-04 r-xp 00000000 00:00 0 {unrelated}\n",
        encoding="utf-8",
    )
    records = loaded_binary_fingerprints(maps)
    assert [Path(record["path"]).name for record in records] == [extension.name, kernel.name]
    assert records[0]["sha256"] == hashlib.sha256(extension.read_bytes()).hexdigest()


def test_cross_stream_pressure_releases_inputs_before_third_stream_writes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests import run_device_acceptance as runner

    log = []
    class Stream:
        def __init__(self, name, npu):
            self.name, self.npu = name, npu
        def record_event(self):
            log.append("event:" + self.name)
            return self.name + "-event"
        def wait_event(self, event):
            log.append("wait:" + self.name + ":" + event)
    class Npu:
        def __init__(self):
            self.current = None
            self.count = 0
        def current_device(self):
            return 0
        def Stream(self):
            stream = Stream(("producer", "compute", "pressure")[self.count % 3], self)
            self.count += 1
            return stream
        def stream(self, stream):
            outer = self
            class Context:
                def __enter__(self):
                    outer.current = stream.name
                def __exit__(self, *args):
                    outer.current = None
            return Context()
        def synchronize(self):
            log.append("synchronize")
    class Tensor:
        def __init__(self, name, input_tensor=False):
            self.name = name
            self.shape = (2, 8)
            self.dtype = "float16"
            self.device = "npu:0"
            self.input_tensor = input_tensor
        def fill_(self, value):
            log.append("write:" + self.name)
            return self
        def __del__(self):
            if self.input_tensor:
                log.append("release:" + self.name)
    npu = Npu()
    torch = SimpleNamespace(npu=npu, empty=lambda *args, **kwargs: Tensor("pressure-block"))
    monkeypatch.setattr(runner, "device_inputs", lambda *args, **kwargs: [Tensor(f"input{i}", input_tensor=True) for i in range(5)])
    monkeypatch.setattr(runner, "verifier_generate", SimpleNamespace(generate=lambda case: []), raising=False)
    monkeypatch.setattr(runner, "verifier_golden", SimpleNamespace(calculate=lambda arrays, attrs: object()))
    class Solution:
        def prepare(self, inputs, attrs):
            return object()
        def launch(self, entry):
            log.append("launch:" + entry + ":" + npu.current)
            return (Tensor("output"), Tensor("output"))
        def run_prepared(self, plan, inputs):
            return self.launch("run")
        def run_prepared_direct(self, plan, inputs):
            return self.launch("run_direct")
    verifier = SimpleNamespace(validate_ownership=lambda inputs, outputs: None,
                               compare=lambda *args: {"passed": True})
    case = {"case_name": "case", "input_desc": [{"shape": [2, 8], "data_type": "float16"}], "attr_desc": []}
    records = runner.run_cross_stream_pressure(torch, Solution(), verifier, case, tmp_path)
    assert [record["entry"] for record in records] == ["run", "run_direct"]
    first_launch = log.index("launch:run:compute")
    first_release = log.index("release:input0")
    first_write = log.index("write:pressure-block")
    assert first_launch < first_release < first_write < log.index("synchronize")
    assert "wait:compute:producer-event" in log[:first_launch]
    assert all(record["pressure_calls"] == 32 for record in records)
