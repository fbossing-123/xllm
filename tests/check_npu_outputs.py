"""Independent FP64 verification for raw operator outputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from tests.generate_npu_cases import from_raw


def _load_raw(directory, name, shape, dtype):
    path = directory / f"{name}.bin"
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != shape[0] * shape[1] if len(shape) == 2 else raw.size != shape[0]:
        raise ValueError(f"{name}.bin has wrong element count")
    return raw.reshape(shape)


def expected_outputs(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    n, h, dtype = metadata["n"], metadata["h"], metadata["dtype"]
    x = from_raw(_load_raw(directory, "x", (n, h), dtype), dtype)
    params = [from_raw(_load_raw(directory, name, (h,), dtype), dtype)
              for name in ("gamma1", "beta1", "gamma2", "beta2")]
    centered = x - x.mean(axis=1, keepdims=True)
    variance = np.mean(centered * centered, axis=1, keepdims=True)
    z = centered / np.sqrt(variance + metadata["epsilon"])
    from tests.generate_npu_cases import to_raw
    expected1 = to_raw(z * params[0] + params[1], dtype)
    expected2 = to_raw(z * params[2] + params[3], dtype)
    return metadata, expected1, expected2


def check_case(directory):
    directory = Path(directory)
    metadata, expected1, expected2 = expected_outputs(directory)
    n, h, dtype = metadata["n"], metadata["h"], metadata["dtype"]
    actual = [_load_raw(directory, name, (n, h), dtype) for name in ("y1", "y2")]
    expected = [expected1, expected2]
    tolerance = 2e-3 if dtype == "float16" else 2e-2
    max_errors = []
    failures = []
    for index, (actual_raw, expected_raw) in enumerate(zip(actual, expected), 1):
        actual_values = from_raw(actual_raw, dtype)
        expected_values = from_raw(expected_raw, dtype)
        max_errors.append(float(np.max(np.abs(actual_values - expected_values))))
        if not np.allclose(actual_values, expected_values, atol=tolerance, rtol=tolerance, equal_nan=True):
            failures.append(f"y{index}")
    if failures:
        raise ValueError(f"accuracy mismatch: {', '.join(failures)}; max_abs={max_errors}")

    exact_beta_rows = 0
    if metadata.get("mode") == "constant" or h == 1:
        beta1 = _load_raw(directory, "beta1", (h,), dtype)
        beta2 = _load_raw(directory, "beta2", (h,), dtype)
        for row in range(n):
            if np.array_equal(actual[0][row], beta1) and np.array_equal(actual[1][row], beta2):
                exact_beta_rows += 1
        if exact_beta_rows != n:
            raise ValueError(f"exact beta check failed for {n - exact_beta_rows} rows")
    return {"passed": True, "n": n, "h": h, "dtype": dtype,
            "max_abs": max(max_errors), "exact_beta_rows": exact_beta_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_case(args.directory)))


if __name__ == "__main__":
    main()
