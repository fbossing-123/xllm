#!/usr/bin/env python3
"""Fail-closed NPU acceptance runner for the supplied DualAffineLayerNorm task."""

import argparse
import copy
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

import numpy as np

try:
    from .device_cases import (
        LIFETIME_PRESSURE_CALLS, LIFETIME_RETAINED_CALLS, MEMORY_RELEASE_LOOPS,
        MIN_VERIFIER_COUNTEREXAMPLES, PERFORMANCE_PROTOCOL, build_manifest, load_coverage,
    )
except ImportError:
    from device_cases import (
        LIFETIME_PRESSURE_CALLS, LIFETIME_RETAINED_CALLS, MEMORY_RELEASE_LOOPS,
        MIN_VERIFIER_COUNTEREXAMPLES, PERFORMANCE_PROTOCOL, build_manifest, load_coverage,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from tools.fingerprint import create_manifest
from tools.task_integrity import authenticate_task_root
from tools.environment_contract import formal_environment_errors

GROUPS = ("accuracy", "extra", "invalid", "stream", "lifetime", "memory", "performance", "verifier")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def case_sha256(case):
    return sha256_bytes(json.dumps(case, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def require_complete_records(records, expected, keys):
    """Require exactly one passing record for every manifest item."""
    expected_keys = [tuple(item[key] for key in keys) for item in expected]
    actual_keys = [tuple(record.get(key) for key in keys) for record in records]
    duplicates = sorted({key for key in actual_keys if actual_keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate evidence records: {duplicates}")
    missing = sorted(set(expected_keys) - set(actual_keys))
    unexpected = sorted(set(actual_keys) - set(expected_keys))
    if missing or unexpected:
        raise ValueError(f"incomplete evidence: missing={missing}, unexpected={unexpected}")
    failed = [record for record in records if record.get("status") != "passed"]
    if failed:
        raise ValueError(f"failed evidence records: {failed}")
    return records


def cann_version(environment):
    info = environment.get("cann_installation_info")
    text = info.get("content", "") if isinstance(info, dict) else ""
    match = re.search(r"(?:version|Version)\s*=\s*([0-9]+(?:\.[0-9]+)+)", text)
    return match.group(1) if match else "unknown"


def read_os_name(path=Path("/etc/os-release")):
    try:
        fields = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                fields[key] = value.strip().strip('"')
        name = fields.get("PRETTY_NAME", "unknown")
    except OSError:
        name = "unknown"
    return " ".join(name.replace("(", " ").replace(")", " ").split())


def query_core_counts(device):
    library = ctypes.CDLL("libascendcl.so")
    counts = []
    for attribute in (201, 102):  # ACL_DEV_ATTR_VECTOR_CORE_NUM, ACL_DEV_ATTR_CUBE_CORE_NUM
        value = ctypes.c_int64(-1)
        result = library.aclrtGetDeviceInfo(device, attribute, ctypes.byref(value))
        if result != 0 or value.value < 0:
            raise RuntimeError(f"aclrtGetDeviceInfo({attribute}) failed with {result}")
        counts.append(value.value)
    return tuple(counts)


def collect_npu_smi():
    completed = subprocess.run(["npu-smi", "info"], text=True, capture_output=True, timeout=30)
    output = completed.stdout + completed.stderr
    match = re.search(r"Version\s*:\s*([^\s|]+)", output)
    return (match.group(1) if match else "unknown"), output


def collect_environment(benchmark, torch, torch_npu, device):
    environment = benchmark.environment_info(torch, torch_npu, device)
    environment["npu_executed"] = True
    environment["status"] = "pending"
    environment["architecture"] = platform.machine().lower()
    environment["os_name"] = read_os_name()
    environment["collection_errors"] = []
    try:
        environment["logical_devices"] = int(torch.npu.device_count())
    except Exception as error:
        environment["logical_devices"] = -1
        environment["collection_errors"].append(f"device_count: {error}")
    try:
        environment["vector_cores"], environment["cube_cores"] = query_core_counts(device)
    except Exception as error:
        environment["vector_cores"] = -1
        environment["cube_cores"] = -1
        environment["collection_errors"].append(f"core_counts: {error}")
    try:
        environment["driver_version"], environment["npu_smi_info"] = collect_npu_smi()
    except Exception as error:
        environment["driver_version"] = "unknown"
        environment["npu_smi_info"] = ""
        environment["collection_errors"].append(f"npu-smi: {error}")
    try:
        import torch_npu.version as torch_npu_version

        environment["torch_npu_git_version"] = str(getattr(torch_npu_version, "git_version", ""))
    except Exception as error:
        environment["torch_npu_git_version"] = ""
        environment["collection_errors"].append(f"torch_npu_git_version: {error}")
    return environment


def validate_environment(environment, require_formal):
    environment = dict(environment)
    environment["cann_version"] = cann_version(environment)
    if not environment.get("npu_executed"):
        environment["status"] = "failed"
        raise ValueError("no NPU was executed; acceptance evidence is incomplete")
    if require_formal:
        problems = formal_environment_errors(environment)
        if problems:
            environment["status"] = "failed"
            environment["formal_environment_errors"] = problems
            raise ValueError("formal environment mismatch: " + "; ".join(problems))
    environment["status"] = "passed"
    environment["formal_required"] = bool(require_formal)
    return environment


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path).resolve())
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_fingerprints(project_root):
    result = {}
    for path in sorted(Path(project_root).rglob("*")):
        if not path.is_file() or any(part in {"build", "artifacts", "__pycache__", ".superpowers"} for part in path.parts):
            continue
        if path.suffix in {".py", ".cpp", ".h", ".cmake", ".md", ".json", ".sh"} or path.name == "CMakeLists.txt":
            result[str(path.relative_to(project_root)).replace("\\", "/")] = sha256_bytes(path.read_bytes())
    return result


def loaded_binary_fingerprints(maps_path=Path("/proc/self/maps")):
    """Hash the implementation libraries actually mapped into this test process."""
    relevant = {"libdual_affine_layer_norm_torch_kernel.so", "libcust_opapi.so"}
    paths = set()
    for line in Path(maps_path).read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = Path(fields[5])
        if path.name in relevant or path.name.startswith("_dual_affine_layer_norm."):
            paths.add(path)
    records = []
    for path in sorted(paths):
        if not path.is_file():
            raise ValueError(f"loaded implementation binary is unavailable for hashing: {path}")
        records.append({"path": str(path.resolve()), "sha256": sha256_bytes(path.read_bytes())})
    return records


def device_inputs(torch, case, arrays, device, offset=0):
    result = []
    for index, (desc, array) in enumerate(zip(case["input_desc"], arrays)):
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[desc["data_type"]]
        raw = array.view(np.int16).reshape(-1)
        padding = offset + index if offset else 0
        if padding:
            raw = np.concatenate((np.zeros(padding, dtype=np.int16), raw))
        flat = torch.from_numpy(raw.copy()).to(device)
        value = flat[padding:].view(dtype).view(desc["shape"])
        if padding and value.storage_offset() == 0:
            raise AssertionError("offset fixture did not create a nonzero storage offset")
        result.append(value)
    return result


def assert_inputs_unchanged(torch, verifier, inputs, arrays):
    for tensor, original in zip(inputs, arrays):
        actual = verifier.cpu_array(tensor).view(np.uint16)
        if not np.array_equal(actual, original.view(np.uint16)):
            raise AssertionError("input changed")


def execute_case(torch, solution, verifier, case, arrays, attrs, entry, offset=0, prepared=False, plan=None):
    inputs = device_inputs(torch, case, arrays, f"npu:{torch.npu.current_device()}", offset)
    expected = verifier_golden.calculate(arrays, attrs)
    if plan is None and prepared:
        plan = solution.prepare(inputs, attrs)
    if entry == "run":
        outputs = solution.run_prepared(plan, inputs) if prepared else solution.run(inputs, attrs)
    elif entry == "run_direct":
        outputs = solution.run_prepared_direct(plan, inputs) if prepared else solution.run_direct(inputs, attrs)
    else:
        raise ValueError(f"unknown execution entry: {entry}")
    outputs = tuple(outputs)
    torch.npu.synchronize()
    verifier.validate_ownership(inputs, outputs)
    checks = verifier.compare(case, arrays, outputs, expected)
    assert_inputs_unchanged(torch, verifier, inputs, arrays)
    return checks


# Set only inside main; keeping golden loading in one place avoids accidental CPU substitutes.
verifier_golden = None


def guarded_record(record, function):
    try:
        details = function()
        record.update(status="passed")
        if details is not None:
            record["details"] = details
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    return record


def run_extra(torch, solution, verifier, cases_by_name, manifest, raw_dir):
    records = []
    for item in manifest["extra"]:
        case_id = item["case_id"]
        base_id, marker = case_id.split("__extra_seed_", 1)
        case = copy.deepcopy(cases_by_name[base_id])
        case["case_name"] = case_id
        arrays = verifier_generate.generate(case)
        attrs = {attr["name"]: attr["value"] for attr in case["attr_desc"]}
        record = dict(item, base_case_id=base_id, seed=int(marker), storage_offsets=True,
                      started_at_utc=utc_now())
        records.append(guarded_record(record, lambda: execute_case(
            torch, solution, verifier, case, arrays, attrs, item["entry"], offset=1 + int(marker) % 4)))
        write_jsonl(raw_dir / "extra.jsonl", records)
    require_complete_records(records, manifest["extra"], ("case_id", "entry"))
    return records


def select_invalid_base_case(public_cases):
    for case in public_cases:
        shape = case["input_desc"][0]["shape"]
        if len(shape) == 2 and all(dimension > 1 for dimension in shape):
            return case
    raise ValueError("no public matrix case can construct a noncontiguous fixture")


def invalid_inputs(torch, solution, scenario, valid, attrs):
    inputs = list(valid)
    attributes = attrs
    if scenario == "affine_dtype":
        inputs[1] = inputs[1].float()
    elif scenario == "affine_length":
        inputs[1] = inputs[1][:-1]
    elif scenario == "attributes_type":
        attributes = []
    elif scenario == "cpu_input":
        inputs[0] = inputs[0].cpu()
    elif scenario == "empty_n":
        inputs[0] = inputs[0][:0]
    elif scenario.startswith("epsilon_"):
        token = scenario[len("epsilon_"):]
        attributes = {"epsilon": {
            "'1e-5'": "1e-5", "-1": -1, "0": 0, "0.1": 0.1,
            "1e-08": 1e-8, "True": True, "inf": float("inf"), "nan": float("nan"),
        }[token]}
    elif scenario == "float32":
        inputs = [tensor.float() for tensor in inputs]
    elif scenario == "input_count":
        inputs = inputs[:-1]
    elif scenario == "large_h":
        n = valid[0].shape[0]
        inputs = [torch.randn((n, 8193), device=valid[0].device, dtype=valid[0].dtype)]
        inputs += [torch.randn(8193, device=valid[0].device, dtype=valid[0].dtype) for _ in range(4)]
    elif scenario == "large_n":
        h = valid[0].shape[1]
        inputs[0] = torch.randn((4097, h), device=valid[0].device, dtype=valid[0].dtype)
    elif scenario == "noncontiguous":
        n, h = valid[0].shape
        inputs[0] = torch.empty((n, h * 2), device=valid[0].device, dtype=valid[0].dtype)[:, ::2]
    elif scenario == "rank":
        inputs[0] = inputs[0].reshape(-1)
    elif scenario == "unknown_attribute":
        attributes = {"unknown": 1}
    else:
        raise ValueError(f"unknown negative scenario: {scenario}")
    return inputs, attributes


def run_invalid(torch, solution, verifier, case, manifest, raw_dir):
    arrays = verifier_generate.generate(case)
    attrs = {attr["name"]: attr["value"] for attr in case["attr_desc"]}
    valid = device_inputs(torch, case, arrays, f"npu:{torch.npu.current_device()}")
    records = []
    for item in manifest["invalid"]:
        def attempt():
            inputs, attributes = invalid_inputs(torch, solution, item["scenario"], valid, attrs)
            getattr(solution, item["entry"])(inputs, attributes)
        record = dict(item, started_at_utc=utc_now())
        try:
            attempt()
        except (TypeError, ValueError, RuntimeError) as error:
            record.update(status="passed", rejected_by=f"{type(error).__name__}: {error}")
        else:
            record.update(status="failed", error="invalid input was accepted")
        records.append(record)
        write_jsonl(raw_dir / "invalid.jsonl", records)
    require_complete_records(records, manifest["invalid"], ("scenario", "entry"))
    return records


def run_stream(torch, solution, verifier, cases_by_name, manifest, raw_dir):
    records = []
    for item in manifest["stream"]:
        case = cases_by_name[item["case_id"]]
        arrays = verifier_generate.generate(case)
        attrs = {attr["name"]: attr["value"] for attr in case["attr_desc"]}
        def exercise():
            stream = torch.npu.Stream()
            details = []
            with torch.npu.stream(stream):
                inputs = device_inputs(torch, case, arrays, f"npu:{torch.npu.current_device()}", offset=3)
                plan = solution.prepare(inputs, attrs)
                for iteration in range(item["prepared_calls"]):
                    entry = "run" if iteration % 2 == 0 else "run_direct"
                    checks = execute_case(torch, solution, verifier, case, arrays, attrs, entry, offset=3, plan=plan)
                    details.append({"iteration": iteration, "entry": entry, "outputs": checks})
                stream.synchronize()
            return details
        records.append(guarded_record(dict(item, started_at_utc=utc_now()), exercise))
        write_jsonl(raw_dir / "stream.jsonl", records)
    require_complete_records(records, manifest["stream"], ("case_id", "stream_index"))
    return records


def run_lifetime(torch, solution, verifier, case, raw_dir):
    arrays = verifier_generate.generate(case)
    attrs = {attr["name"]: attr["value"] for attr in case["attr_desc"]}
    expected = verifier_golden.calculate(arrays, attrs)
    records = []
    for entry in ("run", "run_direct"):
        def exercise():
            retained = []
            inputs = device_inputs(torch, case, arrays, f"npu:{torch.npu.current_device()}", offset=5)
            plan = solution.prepare(inputs, attrs)
            for _ in range(LIFETIME_RETAINED_CALLS):
                outputs = (solution.run_prepared(plan, inputs) if entry == "run"
                           else solution.run_prepared_direct(plan, inputs))
                retained.append(tuple(outputs))
            for _ in range(LIFETIME_PRESSURE_CALLS):
                pressure_inputs = [tensor.clone() for tensor in inputs]
                outputs = (solution.run_prepared(plan, pressure_inputs) if entry == "run"
                           else solution.run_prepared_direct(plan, pressure_inputs))
                del pressure_inputs
                del outputs
                for _ in range(2):
                    pressure = torch.empty_like(inputs[0])
                    pressure.copy_(inputs[0])
                    del pressure
            del inputs
            torch.npu.synchronize()
            gc.collect()
            if len(retained) != LIFETIME_RETAINED_CALLS:
                raise AssertionError("retained output count mismatch")
            storages = [tensor.untyped_storage().data_ptr() for group in retained for tensor in group]
            if len(set(storages)) != len(storages):
                raise AssertionError("retained outputs share storage")
            details = []
            for index, outputs in enumerate(retained):
                details.append({"call": index, "outputs": verifier.compare(case, arrays, outputs, expected)})
            return details
        records.append(guarded_record({"entry": entry, "retained_calls": LIFETIME_RETAINED_CALLS,
                                       "pressure_calls": LIFETIME_PRESSURE_CALLS,
                                       "started_at_utc": utc_now()}, exercise))
        write_jsonl(raw_dir / "lifetime.jsonl", records)
    require_complete_records(records, [{"entry": "run"}, {"entry": "run_direct"}], ("entry",))
    return records


def run_cross_stream_pressure(torch, solution, verifier, case, raw_dir):
    """Release producer inputs after launch, then reuse pressure on an independent stream."""
    arrays = verifier_generate.generate(case)
    attrs = {attr["name"]: attr["value"] for attr in case["attr_desc"]}
    expected = verifier_golden.calculate(arrays, attrs)
    records = []
    for entry in ("run", "run_direct"):
        def exercise():
            producer_stream = torch.npu.Stream()
            compute_stream = torch.npu.Stream()
            pressure_stream = torch.npu.Stream()
            with torch.npu.stream(producer_stream):
                inputs = device_inputs(torch, case, arrays, f"npu:{torch.npu.current_device()}", offset=7)
                producer_event = producer_stream.record_event()
            compute_stream.wait_event(producer_event)
            with torch.npu.stream(compute_stream):
                plan = solution.prepare(inputs, attrs)
                outputs = (solution.run_prepared(plan, inputs) if entry == "run"
                           else solution.run_prepared_direct(plan, inputs))
                outputs = tuple(outputs)
                verifier.validate_ownership(inputs, outputs)
            input_shape, input_dtype = inputs[0].shape, inputs[0].dtype
            input_device = inputs[0].device
            del inputs
            gc.collect()
            pressure_blocks = []
            with torch.npu.stream(pressure_stream):
                for iteration in range(LIFETIME_PRESSURE_CALLS):
                    block = torch.empty(input_shape, dtype=input_dtype, device=input_device)
                    block.fill_(float(iteration + 1))
                    pressure_blocks.append(block)
            torch.npu.synchronize()
            checks = verifier.compare(case, arrays, outputs, expected)
            return {"outputs": checks, "producer_consumer_event_wait": True,
                    "synchronize_before_input_release": False,
                    "pressure_stream_independent": True}
        records.append(guarded_record({
            "case_id": case["case_name"], "entry": entry,
            "pressure_calls": LIFETIME_PRESSURE_CALLS, "early_input_release": True,
            "started_at_utc": utc_now(),
        }, exercise))
        write_jsonl(raw_dir / "cross_stream_lifetime.jsonl", records)
    require_complete_records(records, [{"entry": "run"}, {"entry": "run_direct"}], ("entry",))
    return records


def run_memory(torch, solution, verifier, cases, manifest, raw_dir):
    from dual_affine_layernorm.tiling import make_tiling
    records = []
    for case in cases:
        arrays = verifier_generate.generate(case)
        attrs = {attr["name"]: attr["value"] for attr in case["attr_desc"]}
        inputs = device_inputs(torch, case, arrays, f"npu:{torch.npu.current_device()}")
        n, h = case["input_desc"][0]["shape"]
        dtype = case["input_desc"][0]["data_type"]
        workspace = make_tiling(n, h, dtype=dtype)["workspace_size"]
        output_bytes = 2 * n * h * 2
        for entry in ("run", "run_direct"):
            def exercise():
                call = lambda: getattr(solution, entry)(inputs, attrs)
                outputs = call()
                torch.npu.synchronize()
                del outputs
                gc.collect()
                torch.npu.synchronize()
                baseline = int(torch.npu.memory_allocated())
                torch.npu.reset_peak_memory_stats()
                outputs = call()
                torch.npu.synchronize()
                peak = int(torch.npu.max_memory_allocated())
                delta = peak - baseline
                del outputs
                active_after_release = []
                for _ in range(MEMORY_RELEASE_LOOPS):
                    outputs = call()
                    del outputs
                    torch.npu.synchronize()
                    active_after_release.append(int(torch.npu.memory_allocated()))
                if workspace != 0:
                    raise AssertionError(f"explicit workspace is {workspace}, not zero")
                if delta > output_bytes + 16 * 1024:
                    raise AssertionError(f"active allocation peak delta {delta} exceeds {output_bytes + 16 * 1024}")
                if any(value != baseline for value in active_after_release):
                    raise AssertionError(f"active memory did not return to baseline {baseline}: {active_after_release}")
                return {"workspace_size": workspace, "baseline_active_bytes": baseline,
                        "peak_active_bytes": peak, "peak_delta_bytes": delta,
                        "output_bytes": output_bytes, "allowed_delta_bytes": output_bytes + 16 * 1024,
                        "active_bytes_after_20_releases": active_after_release}
            records.append(guarded_record({"case_id": case["case_name"], "entry": entry,
                                           "input_shape": f"[{n},{h}]", "dtype": dtype,
                                           "release_loops": MEMORY_RELEASE_LOOPS,
                                           "started_at_utc": utc_now()}, exercise))
            write_jsonl(raw_dir / "memory.jsonl", records)
        del inputs
        gc.collect()
        torch.npu.synchronize()
    require_complete_records(records, manifest["memory"], ("case_id", "entry"))
    return records


def run_logged(command, raw_dir, name, cwd, env=None):
    completed = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    (raw_dir / f"{name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (raw_dir / f"{name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    record = {"command": command, "cwd": str(cwd), "exit_code": completed.returncode,
              "stdout_log": f"{name}.stdout.log", "stderr_log": f"{name}.stderr.log",
              "finished_at_utc": utc_now()}
    write_json(raw_dir / f"{name}.command.json", record)
    return completed


def case_metadata(task_root):
    """Map case_name -> (shape string, dtype string) from the supplied case.json."""
    path = Path(task_root) / "case.json"
    if not path.is_file():
        return {}
    cases = json.loads(path.read_text(encoding="utf-8"))
    meta = {}
    for case in cases:
        desc = (case.get("input_desc") or [{}])[0]
        shape = desc.get("shape")
        meta[case["case_name"]] = (
            "[" + ",".join(str(int(dim)) for dim in shape) + "]" if shape is not None else "",
            desc.get("data_type", ""),
        )
    return meta


def shared_fp32_statistics(torch, inputs, epsilon):
    """Task book 3.3 baseline: one FP32 statistics pass plus two affine branches."""
    x, g1, b1, g2, b2 = inputs
    a = x.float()
    centered = a - a.mean(dim=1, keepdim=True)
    z = centered * torch.rsqrt((centered * centered).mean(dim=1, keepdim=True) + epsilon)
    return ((z * g1.float() + b1.float()).to(x.dtype), (z * g2.float() + b2.float()).to(x.dtype))


def run_baselines(torch, benchmark, verifier, cases, raw_dir):
    """Measure both task book baselines on this NPU with the official protocol.

    Each candidate must pass the accuracy check before its timing is kept; the
    faster qualified candidate becomes the per-case baseline and the slower or
    inaccurate one is recorded with its exclusion reason.
    """
    import torch.nn.functional as F

    records = []
    for case in cases:
        arrays = verifier_generate.generate(case)
        attrs = {attr["name"]: attr["value"] for attr in case["attr_desc"]}
        epsilon = attrs.get("epsilon", 1e-5)
        inputs = device_inputs(torch, case, arrays, f"npu:{torch.npu.current_device()}")
        expected = verifier_golden.calculate(arrays, attrs)
        x, g1, b1, g2, b2 = inputs
        candidates = {
            "two_layer_norms": lambda: (F.layer_norm(x, (x.shape[1],), g1, b1, epsilon),
                                        F.layer_norm(x, (x.shape[1],), g2, b2, epsilon)),
            "shared_fp32_statistics": lambda: shared_fp32_statistics(torch, inputs, epsilon),
        }
        entry = {"case_id": case["case_name"], "candidates": []}
        for name, function in candidates.items():
            item = {"name": name, "accuracy": "failed", "status": "failed", "timing": None}
            try:
                outputs = function()
                torch.npu.synchronize()
                verifier.compare(case, arrays, outputs, expected)
                item["accuracy"] = "passed"
                del outputs
                item["timing"] = benchmark.measure(function, torch.npu)
                item["status"] = "passed"
            except Exception as error:
                item["error"] = f"{type(error).__name__}: {error}"
            entry["candidates"].append(item)
        qualified = [item for item in entry["candidates"]
                     if item["accuracy"] == "passed" and item["timing"]]
        if qualified:
            best = min(qualified, key=lambda item: item["timing"]["wall_us_median"])
            entry["selected"] = best["name"]
            entry["baseline_wall_us_median"] = best["timing"]["wall_us_median"]
        else:
            entry["selected"] = None
            entry["baseline_wall_us_median"] = None
        entry["excluded"] = [item["name"] for item in entry["candidates"] if item["name"] != entry["selected"]]
        records.append(entry)
        write_jsonl(raw_dir / "baseline.jsonl", records)
        del inputs
    return records


def run_official(project_root, task_root, solution_file, device, raw_dir,
                 run_accuracy=True, run_performance=True):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    results = []
    if run_accuracy:
        accuracy_report = raw_dir / "public_accuracy.json"
        accuracy = run_logged([
            sys.executable, str(task_root / "verify.py"), "--solution-file", str(solution_file),
            "--device", str(device), "--report", str(accuracy_report),
        ], raw_dir, "public_accuracy", project_root, env)
        results.append(accuracy)
        if accuracy_report.is_file():
            report = json.loads(accuracy_report.read_text(encoding="utf-8"))
            meta = case_metadata(task_root)
            records = []
            for item in report.get("cases", []):
                shape, dtype = meta.get(item["case_name"], ("", ""))
                records.append({**item, "input_shape": shape, "dtype": dtype})
            write_jsonl(raw_dir / "public_accuracy.jsonl", records)
    if run_performance:
        performance_report = raw_dir / "performance.json"
        performance = run_logged([
            sys.executable, str(task_root / "benchmark.py"), "--solution-file", str(solution_file),
            "--device", str(device), "--report", str(performance_report),
        ], raw_dir, "performance", project_root, env)
        acceptance = run_logged([
            sys.executable, str(task_root / "check_performance.py"), "--report", str(performance_report),
            "--targets", str(task_root / "performance_targets.json"), "--cases", str(task_root / "case.json"),
            "--output", str(raw_dir / "performance_acceptance.json"),
        ], raw_dir, "check_performance", project_root, env)
        results.extend((performance, acceptance))
        if performance_report.is_file():
            report = json.loads(performance_report.read_text(encoding="utf-8"))
            meta = case_metadata(task_root)
            records = []
            for item in report.get("cases", []):
                shape, dtype = meta.get(item["case_name"], ("", ""))
                records.append({"case_id": item["case_name"],
                                "status": "passed" if performance.returncode == 0 else "failed",
                                "results": item.get("results"),
                                "input_shape": shape, "dtype": dtype})
            write_jsonl(raw_dir / "performance.jsonl", records)
    if any(result.returncode != 0 for result in results):
        raise RuntimeError("official public accuracy/performance validation failed; inspect command and stderr logs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--require-formal", action="store_true")
    parser.add_argument("--groups", default="all", help="comma-separated groups or all")
    args = parser.parse_args(argv)
    task_root = args.task_root.resolve()
    task_fingerprints = authenticate_task_root(task_root)
    raw_dir = args.output_dir.resolve() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_json(raw_dir / "task_fingerprints.json", task_fingerprints)
    groups = set(GROUPS) if args.groups == "all" else set(args.groups.split(","))
    if not groups or not groups <= set(GROUPS):
        parser.error(f"groups must be all or a subset of {GROUPS}")
    solution_file = PROJECT_ROOT / "solution.py"
    started = utc_now()
    summary = {"status": "running", "started_at_utc": started, "groups": sorted(groups),
               "task_root": str(task_root), "device": args.device, "formal_required": args.require_formal}
    write_json(args.output_dir / "summary.json", summary)
    try:
        import torch
        import torch_npu
        if not torch.npu.is_available():
            raise RuntimeError("NPU unavailable; no acceptance result was produced")
        torch.npu.set_device(args.device)
        sys.path.insert(0, str(task_root))
        global verifier_golden, verifier_generate
        verifier_golden = load_module("acceptance_golden", task_root / "golden.py")
        verifier_generate = load_module("acceptance_generate_data", task_root / "generate_data.py")
        verifier = load_module("acceptance_public_verify", task_root / "verify.py")
        benchmark = load_module("acceptance_public_benchmark", task_root / "benchmark.py")
        solution = load_module("submitted_dual_affine_layernorm", solution_file)
        public_cases = json.loads((task_root / "case.json").read_text(encoding="utf-8"))
        cases_by_name = {case["case_name"]: case for case in public_cases}
        coverage = load_coverage(task_root)
        manifest = build_manifest(coverage)
        for item in manifest["public"]:
            actual = case_sha256(cases_by_name[item["case_id"]])
            if actual != item["input_sha256"]:
                raise ValueError(f"public case fingerprint mismatch: {item['case_id']}")
        write_json(raw_dir / "manifest.json", manifest)
        write_json(raw_dir / "source_fingerprints.json", source_fingerprints(PROJECT_ROOT))
        write_json(raw_dir / "case_fingerprints.json", {
            case["case_name"]: case_sha256(case) for case in public_cases})
        environment = collect_environment(benchmark, torch, torch_npu, args.device)
        write_jsonl(raw_dir / "environment.jsonl", [environment])
        if "accuracy" in groups or "performance" in groups:
            run_official(PROJECT_ROOT, task_root, solution_file, args.device, raw_dir,
                         run_accuracy="accuracy" in groups, run_performance="performance" in groups)
            if "accuracy" in groups:
                records = [dict(json.loads(line)) for line in (raw_dir / "public_accuracy.jsonl").read_text(encoding="utf-8").splitlines()]
                for record in records:
                    record["case_id"] = record["case_name"]
                write_jsonl(raw_dir / "public_accuracy.jsonl", records)
                require_complete_records(records, manifest["public"], ("case_id",))
            if "performance" in groups:
                acceptance = json.loads((raw_dir / "performance_acceptance.json").read_text(encoding="utf-8"))
                if acceptance.get("status") != "passed":
                    raise RuntimeError("official performance acceptance did not pass")
                require_complete_records(
                    [dict(json.loads(line)) for line in (raw_dir / "performance.jsonl").read_text(encoding="utf-8").splitlines()],
                    manifest["performance"], ("case_id",))
                baseline_cases = [cases_by_name[item["case_id"]] for item in manifest["performance"]]
                run_baselines(torch, benchmark, verifier, baseline_cases, raw_dir)
        if "extra" in groups:
            run_extra(torch, solution, verifier, cases_by_name, manifest, raw_dir)
        if "invalid" in groups:
            run_invalid(torch, solution, verifier, select_invalid_base_case(public_cases), manifest, raw_dir)
        if "stream" in groups:
            run_stream(torch, solution, verifier, cases_by_name, manifest, raw_dir)
        if "lifetime" in groups:
            lifetime_case = cases_by_name[manifest["stream"][0]["case_id"]]
            run_lifetime(torch, solution, verifier, lifetime_case, raw_dir)
            run_cross_stream_pressure(torch, solution, verifier, lifetime_case, raw_dir)
        if "memory" in groups:
            memory_cases = [cases_by_name[item["case_id"]] for item in manifest["performance"]]
            run_memory(torch, solution, verifier, memory_cases, manifest, raw_dir)
        if "verifier" in groups:
            try:
                from .verifier_counterexamples import COUNTEREXAMPLE_NAMES, run_counterexamples
            except ImportError:
                from verifier_counterexamples import COUNTEREXAMPLE_NAMES, run_counterexamples
            case = cases_by_name[manifest["stream"][0]["case_id"]]
            arrays = verifier_generate.generate(case)
            attrs = {attr["name"]: attr["value"] for attr in case["attr_desc"]}
            inputs = device_inputs(torch, case, arrays, f"npu:{args.device}")
            outputs = tuple(solution.run(inputs, attrs))
            torch.npu.synchronize()
            expected = verifier_golden.calculate(arrays, attrs)
            records = run_counterexamples(torch, verifier, case, arrays, inputs, expected, outputs,
                                          lambda changed, original: assert_inputs_unchanged(torch, verifier, changed, original))
            write_jsonl(raw_dir / "verifier_counterexamples.jsonl", records)
            if len(COUNTEREXAMPLE_NAMES) < MIN_VERIFIER_COUNTEREXAMPLES:
                raise ValueError("verifier counterexample catalog is below the required minimum")
            require_complete_records(records, [{"name": name} for name in COUNTEREXAMPLE_NAMES], ("name",))
        implementation = create_manifest(PROJECT_ROOT, include_reports=False)
        implementation["loaded_binaries"] = loaded_binary_fingerprints()
        write_json(raw_dir / "implementation_fingerprints.json", implementation)
        validated = validate_environment(environment, args.require_formal)
        write_jsonl(raw_dir / "environment.jsonl", [validated])
        summary.update(status="passed", finished_at_utc=utc_now())
        write_json(args.output_dir / "summary.json", summary)
        return 0
    except Exception as error:
        summary.update(status="failed", finished_at_utc=utc_now(), error=f"{type(error).__name__}: {error}")
        write_json(args.output_dir / "summary.json", summary)
        print(summary["error"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
