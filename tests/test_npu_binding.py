"""Real torch_npu smoke tests; these never substitute a CPU implementation."""

import pytest


torch = pytest.importorskip("torch", reason="PyTorch is required for NPU binding tests")
torch_npu = pytest.importorskip("torch_npu", reason="torch_npu is required for NPU binding tests")

import dual_affine_layernorm as daln


try:
    daln.extension._load_extension()
except RuntimeError as error:
    pytest.skip(str(error), allow_module_level=True)


def make_inputs(n, h, dtype, offset=0):
    generator = torch.Generator(device="cpu").manual_seed(20261029 + n + h + offset)
    values = torch.randn((n * h + offset,), generator=generator, dtype=torch.float32)
    x = values.to(dtype).to("npu")[offset:].view(n, h)
    affine = [torch.randn(h, generator=generator, dtype=torch.float32).to(dtype).to("npu") for _ in range(4)]
    return [x, *affine]


def golden(inputs, epsilon):
    x = inputs[0].cpu().to(torch.float64)
    centered = x - x.mean(dim=1, keepdim=True)
    z = centered / torch.sqrt(centered.square().mean(dim=1, keepdim=True) + epsilon)
    return tuple((z * inputs[i].cpu().to(torch.float64) + inputs[i + 1].cpu().to(torch.float64)).to(inputs[0].dtype)
                 for i in (1, 3))


def check_outputs(actual, expected, dtype):
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    assert actual[0].untyped_storage().data_ptr() != actual[1].untyped_storage().data_ptr()
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output.cpu(), reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n,h", [(1, 1), (3, 17), (49, 33)])
def test_real_npu_both_paths_and_prepared_reuse(dtype, n, h):
    inputs = make_inputs(n, h, dtype, offset=3)
    plan = daln.prepare(inputs, None)
    expected = golden(inputs, 1e-5)
    for actual in [daln.run_prepared(plan, inputs), daln.run_prepared_direct(plan, inputs),
                   daln.run(inputs, None), daln.run_direct(inputs, None)]:
        check_outputs(actual, expected, dtype)
    replacements = make_inputs(n, h, dtype, offset=3)
    for actual in [daln.run_prepared(plan, replacements), daln.run_prepared_direct(plan, replacements)]:
        check_outputs(actual, golden(replacements, 1e-5), dtype)
    torch_npu.npu.synchronize()


def test_prepared_plan_rejects_changed_shape():
    plan = daln.prepare(make_inputs(2, 8, torch.float16), None)
    with pytest.raises(RuntimeError, match="prepared plan"):
        daln.run_prepared(plan, make_inputs(3, 8, torch.float16))
