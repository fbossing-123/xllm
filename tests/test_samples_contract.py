import pytest


SAMPLE_FILES = [
    "kernel_direct/dual_affine_layer_norm_direct.cpp",
    "kernel_direct/dual_affine_layer_norm_direct.h",
    "examples/aclnn_dual_affine_layer_norm.cpp",
    "examples/direct_dual_affine_layer_norm.cpp",
    "examples/sample_common.h",
    "examples/CMakeLists.txt",
]


@pytest.mark.parametrize("relative_path", SAMPLE_FILES)
def test_sample_sources_exist(project_root, relative_path):
    assert (project_root / relative_path).is_file()


def test_direct_launcher_reuses_kernel_without_aclnn(project_root):
    source = (project_root / "kernel_direct/dual_affine_layer_norm_direct.cpp").read_text(encoding="utf-8")
    assert "dual_affine_layer_norm.h" in source
    assert "DualAffineLayerNormTilingData" in source
    assert "<<<" in source
    assert "aclnnDualAffineLayerNorm" not in source
    assert "aclrtMemcpy" not in source


def test_direct_launcher_exposes_concrete_global_kernels_for_stub_generation(project_root):
    source = (project_root / "kernel_direct/dual_affine_layer_norm_direct.cpp").read_text(encoding="utf-8")
    assert "__global__ __aicore__ void dual_affine_layer_norm_direct_fp16(" in source
    assert "__global__ __aicore__ void dual_affine_layer_norm_direct_bf16(" in source
    assert "dual_affine_layer_norm_direct_fp16<<<" in source
    assert "dual_affine_layer_norm_direct_bf16<<<" in source
    assert "dual_affine_layer_norm_direct_kernel<half><<<" not in source


def test_direct_sample_links_tensor_and_launch_runtimes(project_root):
    source = (project_root / "examples/CMakeLists.txt").read_text(encoding="utf-8")
    direct_link = source.split("target_link_libraries(direct_dual_affine_layer_norm", 1)[1]
    assert "libnnopbase.so" in direct_link


def test_aclnn_sample_uses_generated_two_stage_api(project_root):
    source = (project_root / "examples/aclnn_dual_affine_layer_norm.cpp").read_text(encoding="utf-8")
    assert "aclnnDualAffineLayerNormGetWorkspaceSize" in source
    assert "aclnnDualAffineLayerNorm(" in source
    common = (project_root / "examples/sample_common.h").read_text(encoding="utf-8")
    assert "aclDestroyTensor" in common
    assert "aclDestroyAclOpExecutor" in source
    assert "aclrtSynchronizeStream" in source


def test_direct_kernel_header_is_device_compiler_portable(project_root):
    source = (project_root / "kernel_direct/dual_affine_layer_norm_direct.h").read_text(encoding="utf-8")
    assert "#include <acl/acl.h>" not in source
    assert "void* stream" in source


def test_direct_cmake_defines_build_type_for_cann_merge_script(project_root):
    source = (project_root / "examples/CMakeLists.txt").read_text(encoding="utf-8")
    assert 'if(NOT CMAKE_BUILD_TYPE)\n    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)\nendif()' in source
    assert source.index("set(CMAKE_BUILD_TYPE Release") < source.index('include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")')


def test_direct_cmake_initializes_ascendc_language(project_root):
    source = (project_root / "examples/CMakeLists.txt").read_text(encoding="utf-8")
    assert "dual_affine_layer_norm_direct_interface" in source
    assert "ASCEND_CANN_PACKAGE_PATH" in source
    assert "SOC_VERSION" in source
    assert "ascendc.cmake" in source


def test_direct_sample_launches_and_synchronizes(project_root):
    source = (project_root / "examples/direct_dual_affine_layer_norm.cpp").read_text(encoding="utf-8")
    assert "LaunchDualAffineLayerNormDirect" in source
    assert "aclrtSynchronizeStream" in source
    assert "make_tiling" not in source
