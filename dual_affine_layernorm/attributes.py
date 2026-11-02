"""Attribute validation shared by all public Python entry points."""

import math


def parse_attributes(attributes):
    """Return a finite epsilon in the range required by the operator."""
    if attributes is None:
        attributes = {}
    if not isinstance(attributes, dict):
        raise TypeError("attributes must be a dict")
    unknown = set(attributes) - {"epsilon"}
    if unknown:
        raise ValueError(f"unknown attribute: {sorted(unknown)[0]!r}")
    epsilon = attributes.get("epsilon", 1e-5)
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a real number")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or not 1e-7 <= epsilon <= 1e-2:
        raise ValueError("epsilon must be finite and in [1e-7, 1e-2]")
    return epsilon
