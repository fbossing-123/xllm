import numpy as np
import pytest

from tests.check_npu_outputs import check_case
from tests.generate_npu_cases import generate_case


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_binary_fixture_round_trip_and_golden(tmp_path, dtype):
    generate_case(tmp_path, n=3, h=17, dtype=dtype, seed=123)
    metadata, expected1, expected2 = __import__("tests.check_npu_outputs", fromlist=["expected_outputs"]).expected_outputs(tmp_path)
    (tmp_path / "y1.bin").write_bytes(expected1.tobytes())
    (tmp_path / "y2.bin").write_bytes(expected2.tobytes())
    result = check_case(tmp_path)
    assert result["passed"] is True
    assert result["n"] == 3
    assert result["h"] == 17
    assert metadata["dtype"] == dtype


@pytest.mark.parametrize(("dtype", "h"), [("float16", 1), ("bfloat16", 1), ("float16", 33), ("bfloat16", 33)])
def test_constant_and_single_element_rows_are_exact(tmp_path, dtype, h):
    generate_case(tmp_path, n=4, h=h, dtype=dtype, seed=9, mode="constant")
    _, expected1, expected2 = __import__("tests.check_npu_outputs", fromlist=["expected_outputs"]).expected_outputs(tmp_path)
    (tmp_path / "y1.bin").write_bytes(expected1.tobytes())
    (tmp_path / "y2.bin").write_bytes(expected2.tobytes())
    result = check_case(tmp_path)
    assert result["exact_beta_rows"] == 4


def test_checker_rejects_wrong_output_bits(tmp_path):
    generate_case(tmp_path, n=2, h=7, dtype="float16", seed=5)
    _, expected1, expected2 = __import__("tests.check_npu_outputs", fromlist=["expected_outputs"]).expected_outputs(tmp_path)
    wrong = expected1.copy()
    wrong[0, 0] ^= np.uint16(0x1000)
    (tmp_path / "y1.bin").write_bytes(wrong.tobytes())
    (tmp_path / "y2.bin").write_bytes(expected2.tobytes())
    with pytest.raises(ValueError, match="accuracy"):
        check_case(tmp_path)
