"""Push individual changed files into the live A3 session and queue a stage."""

import base64
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
CMD_FILE = PROJECT / "artifacts" / "a3_session" / "cmd.txt"
REMOTE = "/workspace/daln_s1"
RUN = "/workspace/daln_s1_run"
TASK = "/workspace/dual_affine_layernorm_taskdata"

DEFAULT_FILES = [
    "csrc/dual_affine_layer_norm_torch.cpp",
    "tools/remote_round.sh",
]


def send(line: str) -> None:
    with CMD_FILE.open("a", encoding="utf-8", newline="") as handle:
        handle.write(line + chr(10))


def push_file(relative: str) -> None:
    data = (PROJECT / relative).read_bytes().replace(b"\r\n", b"\n")
    payload = base64.b64encode(data).decode("ascii")
    target = f"{REMOTE}/{relative}"
    send(f"cat > {target}.b64 <<'DALN_FILE_EOF'")
    for index in range(0, len(payload), 50000):
        send(payload[index:index + 50000])
    send("DALN_FILE_EOF")
    send(f"base64 -d {target}.b64 > {target} && rm -f {target}.b64 && echo PATCHED:{relative}")


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "perf_auto,perf_off,perf_force,summary"
    rebuild = "--rebuild" in sys.argv
    files = [arg for arg in sys.argv[2:] if not arg.startswith("--")] or DEFAULT_FILES
    for relative in files:
        push_file(relative)
        print(f"queued {relative}")
    if rebuild:
        send(
            f"cd {REMOTE} && export DUAL_AFFINE_LAYERNORM_SOC=ascend910_9382 && "
            f"export ASCEND_CUSTOM_OPP_PATH={REMOTE}/build/packages/vendors/dual_affine_layernorm_custom && "
            f"python setup.py build_ext --inplace > {RUN}/extension_build.log 2>&1; echo EXT_BUILD_RC=$?; "
            f"python -c \"import torch, torch_npu, dual_affine_layernorm as d; d.extension._load_extension(); print('EXT_LOAD_OK')\"")
    send(f"cd {REMOTE} && bash tools/remote_round.sh {REMOTE} {TASK} {RUN} {stage}")
    print("queued stage", stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())