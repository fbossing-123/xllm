"""Device checks for the ACLNN invocation cache (repeated identical calls)."""

import pytest


torch = pytest.importorskip("torch", reason="PyTorch is required for NPU cache tests")
torch_npu = pytest.importorskip("torch_npu", reason="torch_npu is required for NPU cache tests")

import dual_affine_layernorm as daln


try:
    daln.extension._load_extension()
except RuntimeError as error:
    pytest.skip(str(error), allow_module_level=True)


def make_inputs(n, h, dtype, seed=20261030):
    generator = torch.Generator(device="cpu").manual_seed(seed + n * 131 + h)
    values = torch.randn((n * h,), generator=generator, dtype=torch.float32)
    x = values.to(dtype).to("npu").view(n, h)
    affine = [torch.randn(h, generator=generator, dtype=torch.float32).to(dtype).to("npu") for _ in range(4)]
    return [x, *affine]


def golden(inputs, epsilon=1e-5):
    x = inputs[0].cpu().to(torch.float64)
    centered = x - x.mean(dim=1, keepdim=True)
    z = centered / torch.sqrt(centered.square().mean(dim=1, keepdim=True) + epsilon)
    return tuple((z * inputs[i].cpu().to(torch.float64) + inputs[i + 1].cpu().to(torch.float64)).to(inputs[0].dtype)
                 for i in (1, 3))


def tolerate(dtype):
    return 2e-3 if dtype == torch.float16 else 2e-2


def assert_close(outputs, reference, dtype):
    for output, expected in zip(outputs, reference):
        torch.testing.assert_close(output.cpu(), expected, atol=tolerate(dtype), rtol=tolerate(dtype))


def assert_bitwise(outputs, reference):
    for output, expected in zip(outputs, reference):
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16)), "reused executor changed the result"


def copy_outputs(outputs):
    return tuple(output.clone() for output in outputs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n,h", [(1, 768), (7, 33), (49, 1000)])
def test_repeated_calls_are_bitwise_stable(dtype, n, h):
    inputs = make_inputs(n, h, dtype)
    reference = golden(inputs)
    first = daln.run(inputs, {"epsilon": 1e-5})
    torch_npu.npu.synchronize()
    assert_close(first, reference, dtype)
    saved = copy_outputs(first)
    del first
    for _ in range(16):
        outputs = daln.run(inputs, {"epsilon": 1e-5})
        torch_npu.npu.synchronize()
        assert_bitwise(outputs, saved)
        del outputs


def test_altered_output_addresses_stay_correct():
    inputs = make_inputs(64, 257, torch.float16)
    reference = golden(inputs)
    for _ in range(8):
        # Same size class as the outputs, so the allocator is likely to move them.
        padding = [torch.empty((64 * 257,), dtype=torch.float16, device="npu") for _ in range(2)]
        outputs = daln.run(inputs, {"epsilon": 1e-5})
        torch_npu.npu.synchronize()
        assert_close(outputs, reference, torch.float16)
        del outputs, padding


def test_alternating_shapes_hit_and_miss_cleanly():
    shapes = [(1, 768), (128, 4096), (1, 768), (128, 4096)]
    for n, h in shapes:
        inputs = make_inputs(n, h, torch.bfloat16, seed=7)
        outputs = daln.run(inputs, {"epsilon": 1e-5})
        torch_npu.npu.synchronize()
        assert_close(outputs, golden(inputs), torch.bfloat16)


def test_cache_release_keeps_the_public_entry_usable():
    inputs = make_inputs(9, 1537, torch.float16)
    reference = golden(inputs)
    daln.run(inputs, {"epsilon": 1e-5})
    daln.release_cache()
    outputs = daln.run(inputs, {"epsilon": 1e-5})
    torch_npu.npu.synchronize()
    assert_close(outputs, reference, torch.float16)


def test_cache_status_is_reportable():
    status = daln.cache_status()
    assert set(status) >= {"mode", "repeatable_symbol", "active", "hits", "misses"}
    assert status["mode"] in {"auto", "off", "force"}