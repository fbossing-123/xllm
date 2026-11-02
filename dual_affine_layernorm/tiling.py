"""Integer tiling contract mirrored by the CANN host and direct launcher."""


def make_tiling(n, h, core_num=48, ub_size=196608, dtype="float16"):
    for name, value in (("n", n), ("h", h), ("core_num", core_num), ("ub_size", ub_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not 1 <= n <= 4096:
        raise ValueError("n must be in [1, 4096]")
    if not 1 <= h <= 8192:
        raise ValueError("h must be in [1, 8192]")
    if core_num < 1:
        raise ValueError("core_num must be positive")
    if ub_size < 1:
        raise ValueError("ub_size must be positive")
    keys = {"float16": 0, "bfloat16": 1}
    if dtype not in keys:
        raise ValueError("dtype must be float16 or bfloat16")

    aligned_h = (h + 63) // 64 * 64
    # Rows are split across cores when there are more cores than rows; every
    # core of a row recomputes the same full-row statistics, so no workspace
    # and no cross-core reduction is needed.
    min_slice = 512
    cores_per_row = 1
    if n < core_num:
        wanted = max(1, h // min_slice)
        cores_per_row = max(1, min(core_num // n, wanted))
    if cores_per_row > 1:
        slice_h = (h + cores_per_row - 1) // cores_per_row
        slice_h = (slice_h + 63) // 64 * 64
        cores_per_row = (h + slice_h - 1) // slice_h
        block_dim = n * cores_per_row
        rows_per_core = 1
    else:
        cores_per_row = 1
        slice_h = h
        block_dim = min(n, core_num)
        rows_per_core = (n + block_dim - 1) // block_dim
        block_dim = (n + rows_per_core - 1) // rows_per_core
    slice_aligned = (slice_h + 63) // 64 * 64
    ub_bytes = 14 * aligned_h + 8 * slice_aligned
    if ub_bytes > ub_size:
        raise ValueError(f"UB budget exceeded: {ub_bytes} > {ub_size}")
    return {
        "n": n,
        "h": h,
        "aligned_h": aligned_h,
        "rows_per_core": rows_per_core,
        "block_dim": block_dim,
        "cores_per_row": cores_per_row,
        "slice_h": slice_h,
        "slice_aligned": slice_aligned,
        "ub_bytes": ub_bytes,
        "workspace_size": 0,
        "tiling_key": keys[dtype],
    }
