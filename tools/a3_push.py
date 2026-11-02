"""Upload the project sources into the persistent A3 session and run a stage."""

import base64
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a3_round  # noqa: E402

PROJECT = HERE.parent
REMOTE = "/workspace/daln_s1"
RUN = "/workspace/daln_s1_run"
TASK = "/workspace/dual_affine_layernorm_taskdata"


CMD_FILE = HERE.parent / "artifacts" / "a3_session" / "cmd.txt"


def send(line: str) -> None:
    with CMD_FILE.open("a", encoding="utf-8", newline="") as handle:
        handle.write(line + chr(10))


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "prepare,status,perf_auto"
    with tempfile.TemporaryDirectory(prefix="a3-up") as tmp:
        archive = Path(tmp) / "src.tar.gz"
        a3_round.make_archive(PROJECT, archive)
        payload = base64.b64encode(archive.read_bytes()).decode("ascii")
        print(f"archive {archive.stat().st_size} bytes, base64 {len(payload)} chars")
        send("DALN_EOF")  # close any heredoc left by an interrupted upload
        send("rm -f /workspace/daln_src.b64")
        send("cat > /workspace/daln_src.b64 <<'DALN_EOF'")
        for index in range(0, len(payload), 60000):
            send(payload[index:index + 60000])
        send("DALN_EOF")
        send(
            f"base64 -d /workspace/daln_src.b64 > /workspace/daln_src.tgz && "
            f"rm -f /workspace/daln_src.b64 && mkdir -p {REMOTE} && "
            f"tar -xzf /workspace/daln_src.tgz -C {REMOTE} && echo UPLOAD_OK")
        send(
            f"cd {REMOTE} && bash tools/remote_round.sh {REMOTE} {TASK} {RUN} {stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())