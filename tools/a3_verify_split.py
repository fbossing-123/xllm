"""Device correctness sweep for the split-row kernel."""

import numpy as np
import torch
import torch_npu

import dual_affine_layernorm as daln

EPS = 1e-5


def reference(x, params):
    xc = x.cpu().to(torch.float64).numpy()
    mean = xc.mean(axis=1, keepdims=True)
    centered = xc - mean
    var = (centered * centered).mean(axis=1, keepdims=True)
    z = centered / np.sqrt(var + EPS)
    g1, b1, g2, b2 = [t.cpu().to(torch.float64).numpy() for t in params]
    return [z * g1 + b1, z * g2 + b2]


def check(n, h, dtype, const=False):
    torch.manual_seed(1234 + n * 31 + h)
    if const:
        x = torch.full((n, h), 0.75, dtype=torch.float32, device="npu").to(dtype)
    else:
        x = torch.randn(n, h, dtype=torch.float32, device="npu").to(dtype)
    params = [torch.randn(h, dtype=torch.float32, device="npu").to(dtype) for _ in range(4)]
    outputs = daln.run([x, *params], {"epsilon": EPS})
    torch.npu.synchronize()
    assert outputs[0].device.type == "npu" and outputs[0].device.index == 0
    reference_outputs = reference(x, params)
    tolerance = 2e-3 if dtype == torch.float16 else 2e-2
    for index, (actual, expected) in enumerate(zip(outputs, reference_outputs)):
        got = actual.cpu().to(torch.float64).numpy()
        error = np.abs(got - expected)
        allowed = tolerance + tolerance * np.abs(expected)
        if not np.all(error <= allowed):
            raise AssertionError(f"{n}x{h} {dtype} output {index}: max error {error.max()}")
    if const:
        for index, beta in ((0, params[1]), (1, params[3])):
            expected = beta.cpu().unsqueeze(0).expand(n, h).contiguous()
            if not torch.equal(outputs[index].cpu().view(torch.int16), expected.view(torch.int16)):
                raise AssertionError(f"{n}x{h} {dtype} constant row output {index} is not exactly beta")
    print(f"ok {n}x{h} {dtype} const={const}")


def main():
    cases = [
        (2, 8192), (3, 8191), (1, 8192), (2, 8192), (3, 8192), (1, 768), (128, 4096),
        (512, 8192), (49, 33), (33, 65), (1, 1), (7, 1), (17, 127), (31, 256),
        (65, 65), (9, 1537), (4096, 1), (2, 1024),
    ]
    for n, h in cases:
        for dtype in (torch.float16, torch.bfloat16):
            check(n, h, dtype)
    for n, h in ((3, 8192), (2, 64), (33, 65)):
        check(n, h, torch.float16, const=True)
        check(n, h, torch.bfloat16, const=True)
    print("SPLIT_SWEEP_OK")


if __name__ == "__main__":
    main()