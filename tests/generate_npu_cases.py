"""Generate raw ACLNN/direct-kernel fixtures without PyTorch."""

import argparse
import json
from pathlib import Path

import numpy as np

from dual_affine_layernorm.tiling import make_tiling


def to_raw(values, dtype):
    values = np.asarray(values)
    if dtype == "float16":
        return values.astype(np.float16).view(np.uint16)
    if dtype == "bfloat16":
        bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        return rounded.astype(np.uint16)
    raise ValueError(f"unsupported dtype: {dtype}")


def from_raw(raw, dtype):
    raw = np.asarray(raw, dtype=np.uint16)
    if dtype == "float16":
        return raw.view(np.float16).astype(np.float64)
    if dtype == "bfloat16":
        return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    raise ValueError(f"unsupported dtype: {dtype}")


def generate_case(directory, n, h, dtype, seed, mode="normal", epsilon=1e-5):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    make_tiling(n, h, dtype=dtype)
    rng = np.random.default_rng(seed)
    if mode == "constant":
        x = np.repeat(rng.normal(size=(n, 1)) * 3.0, h, axis=1)
    elif mode == "offset":
        x = rng.normal(size=(n, h)) * 0.01 + 1000.0
    else:
        x = rng.normal(size=(n, h))
    arrays = {
        "x": x,
        "gamma1": rng.normal(size=h),
        "beta1": rng.normal(size=h),
        "gamma2": rng.normal(size=h),
        "beta2": rng.normal(size=h),
    }
    for name, values in arrays.items():
        to_raw(values, dtype).astype("<u2").tofile(directory / f"{name}.bin")
    metadata = {"n": n, "h": h, "dtype": dtype, "epsilon": epsilon, "mode": mode, "seed": seed}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], required=True)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--mode", choices=["normal", "constant", "offset"], default="normal")
    parser.add_argument("--epsilon", type=float, default=1e-5)
    args = parser.parse_args()
    print(json.dumps(generate_case(args.directory, args.n, args.h, args.dtype, args.seed, args.mode, args.epsilon)))


if __name__ == "__main__":
    main()
