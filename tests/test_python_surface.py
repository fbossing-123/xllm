import importlib

import pytest


@pytest.fixture(autouse=True)
def reset_extension_handle():
    module = importlib.import_module("dual_affine_layernorm")
    module._EXTENSION = None
    yield


@pytest.fixture
def surface(monkeypatch):
    module = importlib.import_module("dual_affine_layernorm")
    calls = []

    class FakeExtension:
        def prepare(self, inputs, epsilon):
            calls.append(("prepare", inputs, epsilon))
            return "capsule"

        def run_aclnn(self, inputs, epsilon):
            calls.append(("aclnn", inputs, epsilon))
            return "acl outputs"

        def run_direct(self, inputs, epsilon):
            calls.append(("direct", inputs, epsilon))
            return "direct outputs"

        def run_aclnn_prepared(self, capsule, inputs):
            calls.append(("aclnn_prepared", capsule, inputs))
            return "acl outputs"

        def run_direct_prepared(self, capsule, inputs):
            calls.append(("direct_prepared", capsule, inputs))
            return "direct outputs"

        def cache_status(self):
            return {"active": False}

        def release_cache(self):
            calls.append(("release_cache",))

    monkeypatch.setattr(module.extension, "_load_extension", lambda: FakeExtension())
    return module, calls


def test_public_signatures_and_default_epsilon(surface):
    module, calls = surface
    inputs = [object()] * 5
    assert module.run(inputs, None) == "acl outputs"
    assert calls == [("aclnn", inputs, 1e-5)]


def test_hot_path_forwards_validated_epsilon(surface):
    module, calls = surface
    inputs = [object()] * 5
    assert module.run(inputs, {"epsilon": 1e-6}) == "acl outputs"
    assert calls == [("aclnn", inputs, 1e-6)]


def test_unusual_attributes_take_the_validated_path(surface):
    module, calls = surface
    inputs = [object()] * 5
    with pytest.raises(ValueError, match="unknown attribute"):
        module.run(inputs, {"axis": -1})
    assert calls == []


def test_run_direct_uses_distinct_kernel_entry(surface):
    module, calls = surface
    inputs = [object()] * 5
    assert module.run_direct(inputs, {"epsilon": 1e-6}) == "direct outputs"
    assert calls == [("direct", inputs, 1e-6)]


def test_prepare_reuses_opaque_plan_without_retaining_inputs(surface):
    module, calls = surface
    first = [object()] * 5
    second = [object()] * 5
    plan = module.prepare(first, {})
    assert module.run_prepared(plan, second) == "acl outputs"
    assert module.run_prepared_direct(plan, second) == "direct outputs"
    assert calls[0] == ("prepare", first, 1e-5)
    assert calls[1:] == [("aclnn_prepared", "capsule", second), ("direct_prepared", "capsule", second)]
    assert not hasattr(plan, "inputs")


@pytest.mark.parametrize("entry", ["prepare", "run", "run_direct"])
def test_public_entries_reject_wrong_arity_before_extension(surface, entry):
    module, calls = surface
    with pytest.raises(ValueError, match="five tensors"):
        getattr(module, entry)([object()] * 4, None)
    assert calls == []


def test_prepared_entries_reject_foreign_objects(surface):
    module, _ = surface
    with pytest.raises(TypeError, match="PreparedPlan"):
        module.run_prepared(object(), [object()] * 5)
    with pytest.raises(TypeError, match="PreparedPlan"):
        module.run_prepared_direct("plan", [object()] * 5)


def test_missing_extension_has_actionable_error(monkeypatch):
    module = importlib.import_module("dual_affine_layernorm")

    def missing(*args, **kwargs):
        raise ModuleNotFoundError("not built")

    monkeypatch.setattr(module.extension.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="setup.py build_ext --inplace"):
        module.run([object()] * 5, None)


def test_release_cache_is_idempotent(surface):
    module, calls = surface
    assert module.cache_status() == {"active": False}
    module.release_cache()
    module.release_cache()
    assert calls == [("release_cache",), ("release_cache",)]