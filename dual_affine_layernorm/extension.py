"""Lazy loading for the device-only C++ extension."""

import importlib
from functools import lru_cache


@lru_cache(maxsize=1)
def _load_extension():
    try:
        return importlib.import_module("dual_affine_layernorm._dual_affine_layer_norm")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "DualAffineLayerNorm NPU extension is unavailable; build it on the target "
            "host with `python setup.py build_ext --inplace` after installing the "
            "private operator package"
        ) from error
