import pytest

from dual_affine_layernorm.tiling import make_tiling


@pytest.mark.parametrize(
    ("h", "aligned_h"),
    [(1, 64), (15, 64), (16, 64), (17, 64), (31, 64), (33, 64), (64, 64), (65, 128), (8191, 8192), (8192, 8192)],
)
def test_alignment(h, aligned_h):
    assert make_tiling(1, h)["aligned_h"] == aligned_h


def test_maximum_ub_budget_and_zero_workspace():
    result = make_tiling(4096, 8192)
    assert result["ub_bytes"] == 22 * 8192 == 180224
    assert result["workspace_size"] == 0


def test_rows_are_balanced_across_complete_cores():
    result = make_tiling(49, 128)
    assert result["block_dim"] == 25
    assert result["rows_per_core"] == 2
    assert (result["rows_per_core"] - 1) * result["block_dim"] < 49
    assert result["block_dim"] * result["rows_per_core"] >= 49


def test_active_cores_are_recomputed_after_row_rounding():
    result = make_tiling(49, 128)
    assert result["rows_per_core"] == 2
    assert result["block_dim"] == 25
    assert (result["block_dim"] - 1) * result["rows_per_core"] < 49


@pytest.mark.parametrize(("n", "cores"), [(1, 48), (7, 48), (48, 48), (4096, 48), (10, 4)])
def test_block_dim_never_exceeds_rows_or_cores(n, cores):
    assert make_tiling(n, 1, core_num=cores)["block_dim"] == min(n, cores)


def test_dtype_controls_tiling_key():
    assert make_tiling(2, 3, dtype="float16")["tiling_key"] == 0
    assert make_tiling(2, 3, dtype="bfloat16")["tiling_key"] == 1


def test_ub_limit_checks_actual_allocation_without_unused_reserve():
    # One core per row needs 22*aligned_h bytes; H=8192 must fit 180224 bytes.
    assert make_tiling(4096, 8192)["ub_bytes"] == 180224
    with pytest.raises(ValueError, match="UB"):
        make_tiling(4096, 8192, ub_size=180223)


def test_large_rows_are_split_across_spare_cores():
    result = make_tiling(2, 8192)
    assert result["cores_per_row"] == 16
    assert result["slice_h"] == 512
    assert result["slice_aligned"] == 512
    assert result["block_dim"] == 32
    assert result["cores_per_row"] * result["slice_h"] >= 8192
    assert result["ub_bytes"] == 14 * 8192 + 8 * 512
    # Rows that already exceed the core count keep one core per row.
    assert make_tiling(128, 4096)["cores_per_row"] == 1
    assert make_tiling(2, 512)["cores_per_row"] == 1


@pytest.mark.parametrize(("n", "h"), [(0, 1), (4097, 1), (1, 0), (1, 8193)])
def test_shape_limits(n, h):
    with pytest.raises(ValueError):
        make_tiling(n, h)


def test_tiling_has_stable_serializable_fields():
    result = make_tiling(3, 17)
    assert result == {
        "n": 3,
        "h": 17,
        "aligned_h": 64,
        "rows_per_core": 1,
        "block_dim": 3,
        "cores_per_row": 1,
        "slice_h": 17,
        "slice_aligned": 64,
        "ub_bytes": 22 * 64,
        "workspace_size": 0,
        "tiling_key": 0,
    }
