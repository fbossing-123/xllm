from pathlib import Path

import pytest


REQUIRED_SOURCES = [
    "op_kernel/dual_affine_layer_norm.cpp",
    "op_kernel/dual_affine_layer_norm.h",
    "op_kernel/dual_affine_layer_norm_tiling_data.h",
    "op_host/dual_affine_layer_norm_def.cpp",
    "op_host/dual_affine_layer_norm_proto.cpp",
    "op_host/dual_affine_layer_norm_tiling.cpp",
    "op_host/CMakeLists.txt",
    "op_kernel/CMakeLists.txt",
    "CMakeLists.txt",
    "build.sh",
    "install_private.sh",
]


@pytest.mark.parametrize("relative_path", REQUIRED_SOURCES)
def test_standalone_sources_exist(project_root, relative_path):
    assert (project_root / relative_path).is_file(), relative_path


def test_kernel_uses_shared_two_pass_fp32_statistics(project_root):
    source = (project_root / "op_kernel/dual_affine_layer_norm.h").read_text(encoding="utf-8")
    assert source.count("ReduceSum(") >= 2
    assert "Adds(z" in source
    assert "Mul(tmp0, z, z" in source
    assert "sum / static_cast<float>(h_)" in source
    assert "squareSum / static_cast<float>(h_)" in source
    assert "Sqrt(" in source
    assert "E[x^2]" not in source


def test_kernel_handles_unaligned_rows_and_independent_outputs(project_root):
    source = (project_root / "op_kernel/dual_affine_layer_norm.h").read_text(encoding="utf-8")
    assert "DataCopyPad(" in source
    assert "DataCopyExtParams" in source
    assert "y1Gm" in source
    assert "y2Gm" in source
    assert "Muls(z, z, rstd" in source


def test_kernel_output_never_aliases_next_row_input(project_root):
    source = (project_root / "op_kernel/dual_affine_layer_norm.h").read_text(encoding="utf-8")
    assert "LocalTensor<T> out16 = tmp1.ReinterpretCast<T>();" in source
    assert "ComputeBranch(g1, b1, out16" in source
    assert "ComputeBranch(g2, b2, out16" in source
    assert "CopyOut(y1Gm_, out16" in source
    assert "CopyOut(y2Gm_, out16" in source
    assert "ComputeBranch(g1, b1, x16" not in source
    assert "ComputeBranch(g2, b2, x16" not in source


def test_kernel_dispatches_only_fp16_and_bf16(project_root):
    source = (project_root / "op_kernel/dual_affine_layer_norm.cpp").read_text(encoding="utf-8")
    assert "DualAffineLayerNorm<half>" in source
    assert "DualAffineLayerNorm<bfloat16_t>" in source
    assert "DualAffineLayerNorm<float>" not in source
    assert "GET_TILING_DATA_WITH_STRUCT" in source


def test_opdef_contract(project_root):
    source = (project_root / "op_host/dual_affine_layer_norm_def.cpp").read_text(encoding="utf-8")
    for name in ("x", "gamma1", "beta1", "gamma2", "beta2"):
        assert f'Input("{name}")' in source
    assert source.count("Output(\"y") == 2
    assert 'Attr("epsilon")' in source
    assert "AttrType(OPTIONAL)" in source
    assert "Float(1e-5)" in source
    assert "ge::DT_FLOAT16" in source
    assert "ge::DT_BF16" in source
    assert "ge::DT_FLOAT," not in source
    assert 'AddConfig("ascend910_93")' in source
    assert 'AddConfig("ascend910b")' in source


def test_proto_copies_shape_and_dtype_to_both_outputs(project_root):
    source = (project_root / "op_host/dual_affine_layer_norm_proto.cpp").read_text(encoding="utf-8")
    assert source.count("GetOutputShape(") >= 2
    assert source.count("SetOutputDataType(") == 2
    assert "IMPL_OP_INFERSHAPE(DualAffineLayerNorm)" in source


def test_proto_uses_only_registry_headers_available_to_opbuild(project_root):
    source = (project_root / "op_host/dual_affine_layer_norm_proto.cpp").read_text(encoding="utf-8")
    assert '#include "error/ops_error.h"' not in source
    assert '#include "register/op_impl_registry.h"' in source


def test_host_tiling_has_exact_ub_and_zero_workspace(project_root):
    source = (project_root / "op_host/dual_affine_layer_norm_tiling.cpp").read_text(encoding="utf-8")
    assert "14 * alignedH + 8 * sliceAligned" in source
    assert "64" in source
    assert "ubBytes > static_cast<int64_t>(ubSize)" in source
    assert "16 * 1024" not in source
    assert "currentWorkspace[0] = 0" in source
    assert "coresPerRow" in source
    assert "rowsPerCore" in source
    assert "GetCoreNumAiv" in source
    assert "GetCoreMemSize" in source
