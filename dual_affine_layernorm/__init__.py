"""DualAffineLayerNorm public Python and PyTorch NPU interfaces."""

import atexit

from . import extension
from .attributes import parse_attributes
from .tiling import make_tiling

_EPSILON_MIN = 1e-7
_EPSILON_MAX = 1e-2
_DEFAULT_EPSILON = 1e-5
_EXTENSION = None


class PreparedPlan:
    """Opaque immutable metadata; input and output tensors are never retained."""

    __slots__ = ("_capsule",)

    def __init__(self, capsule):
        self._capsule = capsule


def _loaded_extension():
    global _EXTENSION
    loaded = _EXTENSION
    if loaded is None:
        loaded = _EXTENSION = extension._load_extension()
    return loaded


def _check_inputs(inputs):
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 5:
        raise ValueError("inputs must contain exactly five tensors")
    return inputs


def prepare(inputs, attributes=None):
    inputs = _check_inputs(inputs)
    epsilon = parse_attributes(attributes)
    return PreparedPlan(extension._load_extension().prepare(inputs, epsilon))


def run_prepared(plan, inputs):
    if not isinstance(plan, PreparedPlan):
        raise TypeError("plan must be a PreparedPlan returned by prepare")
    inputs = _check_inputs(inputs)
    return extension._load_extension().run_aclnn_prepared(plan._capsule, inputs)


def run_prepared_direct(plan, inputs):
    if not isinstance(plan, PreparedPlan):
        raise TypeError("plan must be a PreparedPlan returned by prepare")
    inputs = _check_inputs(inputs)
    return extension._load_extension().run_direct_prepared(plan._capsule, inputs)


def _run_validated(entry, inputs, attributes):
    inputs = _check_inputs(inputs)
    epsilon = parse_attributes(attributes)
    return entry(inputs, epsilon)


def run(inputs, attributes=None):
    # The hot path keeps the same validation but avoids rebuilding helper
    # containers; anything unusual falls through to the fully validated path.
    global _EXTENSION
    loaded = _EXTENSION
    if loaded is None:
        loaded = _EXTENSION = extension._load_extension()
    if type(inputs) is list and len(inputs) == 5:
        if attributes is None:
            return loaded.run_aclnn(inputs, _DEFAULT_EPSILON)
        if type(attributes) is dict and len(attributes) == 1:
            epsilon = attributes.get("epsilon")
            if type(epsilon) is float and _EPSILON_MIN <= epsilon <= _EPSILON_MAX:
                return loaded.run_aclnn(inputs, epsilon)
    return _run_validated(loaded.run_aclnn, inputs, attributes)


def run_direct(inputs, attributes=None):
    global _EXTENSION
    loaded = _EXTENSION
    if loaded is None:
        loaded = _EXTENSION = extension._load_extension()
    if type(inputs) is list and len(inputs) == 5:
        if attributes is None:
            return loaded.run_direct(inputs, _DEFAULT_EPSILON)
        if type(attributes) is dict and len(attributes) == 1:
            epsilon = attributes.get("epsilon")
            if type(epsilon) is float and _EPSILON_MIN <= epsilon <= _EPSILON_MAX:
                return loaded.run_direct(inputs, epsilon)
    return _run_validated(loaded.run_direct, inputs, attributes)


def cache_status():
    """Report the ACLNN reuse cache state for diagnostics and benchmarks."""
    return _loaded_extension().cache_status()


def release_cache():
    """Drop cached ACLNN handles before the interpreter tears the runtime down."""
    global _EXTENSION
    if _EXTENSION is not None:
        _EXTENSION.release_cache()


atexit.register(release_cache)


__all__ = [
    "PreparedPlan", "cache_status", "make_tiling", "parse_attributes", "prepare", "release_cache",
    "run", "run_direct", "run_prepared", "run_prepared_direct",
]