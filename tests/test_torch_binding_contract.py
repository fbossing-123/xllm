import pytest


def test_binding_source_exists(project_root):
    assert (project_root / "csrc/dual_affine_layer_norm_torch.cpp").is_file()


def test_binding_validates_full_tensor_contract(project_root):
    source = (project_root / "csrc/dual_affine_layer_norm_torch.cpp").read_text(encoding="utf-8")
    for text in [
        "five tensors", "rank-2", "float16 or bfloat16", "PrivateUse1",
        "contiguous", "same device", "same dtype", "N must be", "H must be",
        "affine parameters", "epsilon", "std::isfinite",
    ]:
        assert text in source


def test_binding_allocates_independent_outputs_and_records_input_storage(project_root):
    source = (project_root / "csrc/dual_affine_layer_norm_torch.cpp").read_text(encoding="utf-8")
    assert "EmptyOutput(inputs[0])" in source
    assert "at::detail::empty_generic" in source
    assert "NPUCachingAllocator::get()" in source
    assert "auto y1" in source and "auto y2" in source
    assert "getCurrentNPUStream" in source
    assert "NPUCachingAllocator::recordStream" in source
    assert "for (const auto& input : inputs)" in source
    assert "workspaceSize != 0" in source
    assert "LaunchDualAffineLayerNormDirect" in source
    assert "aclnnDualAffineLayerNormGetWorkspaceSize" in source


def test_prepared_plan_has_no_tensor_members(project_root):
    source = (project_root / "csrc/dual_affine_layer_norm_torch.cpp").read_text(encoding="utf-8")
    plan = source.split("struct Plan {", 1)[1].split("};", 1)[0]
    assert "Tensor" not in plan
    assert "Storage" not in plan
    assert "py::" not in plan


def test_binding_cmake_uses_real_kernel_and_generated_aclnn(project_root):
    source = (project_root / "csrc/CMakeLists.txt").read_text(encoding="utf-8")
    for text in [
        "ascendc_library", "dual_affine_layer_norm_direct.cpp", "cust_opapi",
        "torch_npu", "libascendcl.so", "libnnopbase.so", "SOC_VERSION",
    ]:
        assert text in source


def test_binding_reuses_only_address_validated_executors(project_root):
    source = (project_root / "csrc/dual_affine_layer_norm_torch.cpp").read_text(encoding="utf-8")
    for text in [
        "aclSetAclOpExecutorRepeatable",
        "InvocationKey",
        "inputs[0].data_ptr()",
        "inputs[4].data_ptr()",
        "y1.data_ptr()",
        "y2.data_ptr()",
        "epsilon == other.epsilon",
        "slot.valid && slot.key == key",
        "release_cache",
    ]:
        assert text in source


def test_binding_keeps_per_call_fallback_and_stream_recording(project_root):
    source = (project_root / "csrc/dual_affine_layer_norm_torch.cpp").read_text(encoding="utf-8")
    assert "RunAclnnPerCall" in source
    assert "if (RunAclnnReusing(plan, inputs, y1, y2, stream))" in source
    per_call = source.split("Outputs RunAclnnPerCall", 1)[1]
    assert "aclnnDualAffineLayerNormGetWorkspaceSize" in per_call
    run = source.split("Outputs RunAclnn(", 1)[1].split("Outputs RunDirect", 1)[0]
    assert run.count("RecordInputs(inputs, stream)") == 2


def test_executor_cache_is_not_released_by_static_destruction(project_root):
    source = (project_root / "csrc/dual_affine_layer_norm_torch.cpp").read_text(encoding="utf-8")
    assert "~Invocation() = default;" in source
    assert "~Invocation() {" not in source
    assert "aclDestroyAclOpExecutor" in source
