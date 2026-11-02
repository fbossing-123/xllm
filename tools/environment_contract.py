"""Formal target environment from performance_targets.json.

SoC, CANN, Python, torch and torch_npu must match exactly. The driver is the one
field with a compatibility rule: Ascend drivers are backward compatible, so any
driver at or above the specified 25.5.t6.b001 satisfies the contract.
"""

import re


FORMAL_ENVIRONMENT = {
    "soc": "Ascend910_9382",
    "logical_devices": 1,
    "vector_cores": 48,
    "cube_cores": 24,
    "CANN": "9.1.0",
    "driver": "25.5.t6.b001",
    "architecture": "aarch64",
    "os": "openEuler 24.03 LTS-SP3",
    "python": "3.12.13",
    "torch": "2.10.0+cpu",
    "torch_npu": "2.10.0.post4+git5dd8ef3",
}


def _version_numbers(value):
    return [int(part) for part in re.findall(r"\d+", str(value))]


def driver_satisfies(actual):
    """True when ``actual`` is the required driver or any newer one."""
    required = _version_numbers(FORMAL_ENVIRONMENT["driver"])
    numbers = _version_numbers(actual)
    if not numbers or not required:
        return False
    width = max(len(required), len(numbers))
    required = required + [0] * (width - len(required))
    numbers = numbers + [0] * (width - len(numbers))
    return numbers >= required


def torch_npu_satisfies(actual, git_version):
    """Accept the pinned build, or the same release for the pinned git commit.

    PyPI wheels keep the git commit in ``torch_npu.version.git_version`` instead
    of a ``+git<sha>`` local version label, so the commit prefix is checked
    explicitly rather than only comparing version strings.
    """
    expected = FORMAL_ENVIRONMENT["torch_npu"]
    if actual == expected:
        return True
    if "+git" not in expected:
        return False
    if str(actual).split("+", 1)[0] != expected.split("+", 1)[0]:
        return False
    required_commit = expected.split("+git", 1)[1]
    return str(git_version or "").startswith(required_commit)


def formal_environment_errors(environment):
    required = FORMAL_ENVIRONMENT
    comparisons = (
        ("device_name", required["soc"], "device is not Ascend910_9382"),
        ("cann_version", required["CANN"], "CANN is not 9.1.0"),
        ("python", required["python"], "Python is not 3.12.13"),
        ("torch", required["torch"], "torch is not 2.10.0+cpu"),
        ("architecture", required["architecture"], "architecture is not aarch64"),
        ("os_name", required["os"], "OS is not openEuler 24.03 LTS-SP3"),
        ("logical_devices", required["logical_devices"], "visible logical device count is not 1"),
        ("vector_cores", required["vector_cores"], "Vector Core count is not 48"),
        ("cube_cores", required["cube_cores"], "Cube Core count is not 24"),
        ("device_index", 0, "acceptance device is not logical device 0"),
    )
    problems = [message for field, expected, message in comparisons if environment.get(field) != expected]
    if not torch_npu_satisfies(environment.get("torch_npu"), environment.get("torch_npu_git_version")):
        problems.append(f"torch_npu {environment.get('torch_npu')!r} is not the pinned 2.10.0.post4+git5dd8ef3 build")
    if not driver_satisfies(environment.get("driver_version")):
        problems.append(f"driver {environment.get('driver_version')!r} is older than 25.5.t6.b001")
    if "linux" not in str(environment.get("platform", "")).lower():
        problems.append("host platform is not Linux")
    return problems