"""Independent adversarial checks that the public verifier rejects bad submissions."""


COUNTEREXAMPLE_NAMES = (
    "cpu_output",
    "wrong_device",
    "input_storage_reuse",
    "shared_output_storage",
    "wrong_count",
    "wrong_shape",
    "wrong_dtype",
    "noncontiguous",
    "wrong_values",
    "input_mutation",
)


def _shared_storage_outputs(torch, outputs):
    first = outputs[0]
    storage = torch.empty(first.numel() * 2, dtype=first.dtype, device=first.device)
    shape = tuple(first.shape)
    return storage[:first.numel()].view(shape), storage[first.numel():].view(shape)


def _noncontiguous_output(torch, output):
    n, h = output.shape
    return torch.empty((n, h * 2), dtype=output.dtype, device=output.device)[:, ::2]


def run_counterexamples(torch, verifier, case, arrays, inputs, expected, outputs, assert_inputs_unchanged):
    """Run each sabotage independently; a record passes only when validation raises."""
    checks = {
        "cpu_output": lambda: verifier.compare(case, arrays, tuple(t.cpu() for t in outputs), expected),
        "wrong_device": lambda: verifier.validate_ownership(inputs, tuple(t.cpu() for t in outputs)),
        "input_storage_reuse": lambda: verifier.validate_ownership(inputs, (inputs[0], inputs[1])),
        "shared_output_storage": lambda: verifier.validate_ownership(inputs, _shared_storage_outputs(torch, outputs)),
        "wrong_count": lambda: verifier.compare(case, arrays, outputs[:1], expected),
        "wrong_shape": lambda: verifier.compare(case, arrays, (outputs[0].reshape(-1), outputs[1]), expected),
        "wrong_dtype": lambda: verifier.compare(case, arrays, (outputs[0].float(), outputs[1]), expected),
        "noncontiguous": lambda: verifier.compare(case, arrays, (_noncontiguous_output(torch, outputs[0]), outputs[1]), expected),
        "wrong_values": lambda: verifier.compare(case, arrays, (outputs[0] + 1, outputs[1]), expected),
    }

    def mutated_input():
        mutated = [tensor.clone() for tensor in inputs]
        mutated[0].view(torch.int16).view(-1)[0] ^= 1
        assert_inputs_unchanged(mutated, arrays)

    checks["input_mutation"] = mutated_input
    records = []
    for name in COUNTEREXAMPLE_NAMES:
        try:
            checks[name]()
        except (AssertionError, RuntimeError, ValueError) as error:
            records.append({"name": name, "status": "passed", "rejected_by": str(error)})
        else:
            records.append({"name": name, "status": "failed", "error": "verifier accepted invalid implementation"})
    return records
