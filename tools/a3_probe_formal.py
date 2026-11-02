"""Probe the A3 host for an installable torch 2.10 / torch_npu 2.10 stack."""

import subprocess

CHECKS = [
    ("interpreters", "ls -d /usr/local/python*/ 2>/dev/null; which -a python python3"),
    ("site_torch", "for sp in /usr/local/python*/lib/python3.12/site-packages; do echo \"[$sp]\"; ls \"$sp\" 2>/dev/null | grep -iE '^(torch|torch_npu)' | head -5; done"),
    ("cann_python", "ls /usr/local/Ascend/ascend-toolkit/latest/python/site-packages 2>/dev/null | head -8"),
    ("wheels", "find / -maxdepth 6 \\( -name 'torch_npu*.whl' -o -name 'torch-2.10*.whl' \\) 2>/dev/null | head -10"),
    ("pip_index", "python -m pip index versions torch_npu 2>&1 | head -3"),
    ("net_pytorch", "curl -sS -m 8 -o /dev/null -w '%{http_code}\\n' https://download.pytorch.org/whl/cpu/ 2>&1"),
    ("net_pypi", "curl -sS -m 8 -o /dev/null -w '%{http_code}\\n' https://pypi.org/simple/torch-npu/ 2>&1"),
    ("venv", "python -m venv --help > /dev/null 2>&1 && echo venv_ok || echo venv_missing"),
    ("docker", "command -v docker || command -v podman || echo no_container_runtime"),
    ("docker_images", "docker images 2>/dev/null | head -8"),
    ("python_version", "python -c \"import sys; print(sys.version); print(sys.prefix)\""),
]


def main():
    for name, command in CHECKS:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=180)
            output = (result.stdout or "").strip()
            error = (result.stderr or "").strip()
        except Exception as exc:  # pragma: no cover - diagnostics only
            output, error, result = "", str(exc), None
        print(f"== {name} rc={getattr(result, 'returncode', -1)}")
        if output:
            print(output[:900])
        if error:
            print("ERR", error[:200])


if __name__ == "__main__":
    main()