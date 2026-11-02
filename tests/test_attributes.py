import math

import pytest

from dual_affine_layernorm.attributes import parse_attributes


def test_default_epsilon():
    assert parse_attributes(None) == 1e-5
    assert parse_attributes({}) == 1e-5


@pytest.mark.parametrize("value", [1e-7, 1e-2, 0.001, 1e-6])
def test_valid_epsilon(value):
    assert parse_attributes({"epsilon": value}) == float(value)


@pytest.mark.parametrize("attributes", [[], [("epsilon", 1e-5)], "epsilon=1e-5", 1])
def test_attributes_must_be_a_dictionary(attributes):
    with pytest.raises(TypeError, match="attributes must be a dict"):
        parse_attributes(attributes)


def test_unknown_attributes_are_rejected():
    with pytest.raises(ValueError, match="unknown attribute"):
        parse_attributes({"epsilon": 1e-5, "axis": -1})


@pytest.mark.parametrize("value", [True, False, "1e-5", None, float("nan"), float("inf"), -float("inf"), 0.0, 1e-8, 0.010001])
def test_invalid_epsilon_is_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        parse_attributes({"epsilon": value})


def test_returned_epsilon_is_finite_float():
    result = parse_attributes({"epsilon": 1e-7})
    assert isinstance(result, float)
    assert math.isfinite(result)
