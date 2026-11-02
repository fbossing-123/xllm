#!/usr/bin/env python3
"""Device probe for the public entry: official protocol plus cache soak.

Run it on the target host after building the extension:

    python tools/perf_probe.py \
        --task-root /path/to/supplied_task_package \
        --device 0 --output artifacts/perf_probe.json

The script never modifies the supplied task package; it only reads case.json,
generate_data.py, golden.py, verify.py and benchmark.py from it.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

PROTOCOL = {"warmup": 50, "iterations": 200, "rounds": 5}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def median(values):
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def measure(benchmark, torch, function):
    return benchmark.measure(function, torch.npu)


def soak(torch, solution, verify, case, arrays, inputs, attributes, expected, rounds):
    """Repeat identical calls; every round must match the golden and stay bitwise stable."""
    first = tuple(solution.run(inputs, attributes))
    torch.npu.synchronize()
    verify.compare(case, arrays, first, expected)
    saved = tuple(output.clone() for output in first)
    del first
    for index in range(rounds):
        outputs = tuple(solution.run(inputs, attributes))
        torch.npu.synchronize()
        verify.compare(case, arrays, outputs, expected)
        for output, reference in zip(outputs, saved):
            if not torch.equal(output.view(torch.int16), reference.view(torch.int16)):
                raise AssertionError(f"soak round {index} changed the result")
        del outputs
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure the public entry with the fixed 50/200/5 protocol.")
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--solution", type=Path, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--entries", default="run,run_direct")
    parser.add_argument("--cases", default="")
    parser.add_argument("--soak", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("perf_probe.json"))
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.task_root))
    sys.path.insert(0, str(args.project))
    import torch
    import torch_npu

    import benchmark
    import generate_data
    import verify

    golden = load_module("probed_golden", args.task_root / "golden.py")

    solution_file = args.solution or args.project / "solution.py"
    spec = importlib.util.spec_from_file_location("probed_solution", solution_file.resolve())
    solution = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    if not torch.npu.is_available():
        raise RuntimeError("NPU is unavailable")
    torch.npu.set_device(args.device)

    targets = json.loads((args.task_root / "performance_targets.json").read_text(encoding="utf-8"))
    target_by_case = {entry["case_name"]: entry for entry in targets["cases"]}
    public_cases = json.loads((args.task_root / "case.json").read_text(encoding="utf-8"))
    selected = [case for case in public_cases if case.get("performance") is True]
    if args.cases:
        wanted = set(args.cases.split(","))
        selected = [case for case in selected if case["case_name"] in wanted]

    entries = [entry.strip() for entry in args.entries.split(",") if entry.strip()]
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": dict(PROTOCOL),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_npu": getattr(torch_npu, "__version__", "unknown"),
            "device_name": torch.npu.get_device_name(args.device),
        },
        "cases": [],
        "soak": [],
    }
    try:
        from dual_affine_layernorm import cache_status

        report["cache_status"] = dict(cache_status())
    except Exception as error:  # pragma: no cover - diagnostics only
        report["cache_status"] = f"unavailable: {error}"

    for case in selected:
        name = case["case_name"]
        arrays = generate_data.generate(case)
        attributes = {item["name"]: item["value"] for item in case["attr_desc"]}
        inputs = verify.device_inputs(case, arrays, f"npu:{args.device}")
        expected = golden.calculate(arrays, attributes)
        outputs = tuple(solution.run(inputs, attributes))
        torch.npu.synchronize()
        verify.validate_ownership(inputs, outputs)
        accuracy = verify.compare(case, arrays, outputs, expected)
        del outputs

        record = {"case_name": name, "max_wall_us": target_by_case[name]["max_wall_us"],
                  "accuracy": accuracy, "entries": {}}
        functions = {
            "run": lambda: solution.run(inputs, attributes),
            "run_direct": lambda: solution.run_direct(inputs, attributes),
        }
        if "run_prepared" in entries:
            plan = solution.prepare(inputs, attributes)
            functions["run_prepared"] = lambda: solution.run_prepared(plan, inputs)
        for entry in entries:
            if entry not in functions:
                raise ValueError(f"unsupported entry {entry!r}")
            record["entries"][entry] = measure(benchmark, torch, functions[entry])
        report["cases"].append(record)
        print(json.dumps({"case_name": name,
                          "wall_us": {k: v["wall_us_median"] for k, v in record["entries"].items()},
                          "max_wall_us": record["max_wall_us"]}, ensure_ascii=False), flush=True)
        if args.soak:
            soak(torch, solution, verify, case, arrays, inputs, attributes, expected, args.soak)
            report["soak"].append({"case_name": name, "rounds": args.soak, "status": "passed"})
            print(json.dumps({"case_name": name, "soak": args.soak, "status": "passed"}), flush=True)
        del inputs, expected
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"probe report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())