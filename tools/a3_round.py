#!/usr/bin/env python3
"""Sync the project to the A3 host and run one measurement round in a single SSH session.

The jump token must be passed in A3_JUMP (``jt_...@113.47.8.48:2234``) and the
target password in A3_PASSWORD. Neither value is written to disk or printed. The
token is short-lived, so upload, build and every probe share one connection; the
artifacts are fetched afterwards on a best-effort basis.

    A3_JUMP=... A3_PASSWORD=... python tools/a3_round.py --remote-prefix daln_r5
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

TARGET = "root@199.103.54.86"
TASK_DIR = "/workspace/dual_affine_layernorm_taskdata"
EXCLUDES = (".git", "build", "artifacts", "__pycache__", ".pytest_cache", ".superpowers",
            ".zcode", "rsi", "*.zip", "*.pyc", "*.pyo", ".a3_ssh.py", ".tmp_*", "HANDOFF.html")


def make_archive(project: Path, destination: Path) -> None:
    def keep(info: tarfile.TarInfo):
        parts = Path(info.name).parts
        if "__pycache__" in parts or ".pytest_cache" in parts:
            return None
        if info.name.endswith((".pyc", ".pyo")):
            return None
        if info.name.startswith("dual_affine_layernorm/") and info.name.endswith(".so"):
            return None
        return info

    for script in project.rglob("*.sh"):
        if ".git" in script.parts:
            continue
        if b"\r\n" in script.read_bytes():
            raise SystemExit(f"{script} has CRLF line endings; the remote bash cannot run it")
    with tarfile.open(destination, "w:gz") as archive:
        for entry in sorted(project.iterdir()):
            if any(entry.match(pattern) for pattern in EXCLUDES):
                continue
            archive.add(entry, arcname=entry.name, filter=keep)


class SSH:
    """One password-authenticated, jump-hosted ssh client."""

    def __init__(self, jump: str, password: str):
        self.jump = jump
        self.directory = Path(tempfile.mkdtemp(prefix="a3-driver"))
        password_file = self.directory / "p"
        password_file.write_text(password + "\n", encoding="utf-8")
        os.chmod(password_file, 0o600)
        askpass = self.directory / "askpass.sh"
        askpass.write_text("#!/bin/sh\ncat '%s'\n" % str(password_file).replace("'", "'\\''"),
                           encoding="utf-8")
        os.chmod(askpass, 0o700)
        self.env = dict(os.environ)
        self.env.update({"SSH_ASKPASS": str(askpass), "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": ":0",
                         # Git's ssh is an MSYS binary; without this it rewrites remote
                         # POSIX paths into Windows paths before they leave the host.
                         "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"})

    def argv(self, remote: str):
        return [
            shutil.which("ssh"), "-T",
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=40", "-o", "TCPKeepAlive=yes",
            "-o", "ConnectTimeout=30", "-o", "NumberOfPasswordPrompts=1",
            "-o", "PreferredAuthentications=password,keyboard-interactive", "-o", "PubkeyAuthentication=no",
            "-J", self.jump, TARGET, remote,
        ]

    def run(self, remote: str, *, stdin=None, stdout=None, timeout: int = 600, check=False):
        return subprocess.run(self.argv(remote), env=self.env, stdin=stdin, stdout=stdout,
                              timeout=timeout, check=check)

    def close(self):
        shutil.rmtree(self.directory, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="One A3 stage in one SSH session")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--remote-prefix", default="daln_round")
    parser.add_argument("--reuse-dir", default="",
                        help="existing remote project dir without the leading slash")
    parser.add_argument("--stage", default="all")
    parser.add_argument("--task-dir", default=TASK_DIR)
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if "/" in args.remote_prefix or "\\" in args.remote_prefix:
        print("remote prefix must be a plain directory name", file=sys.stderr)
        return 2
    allowed = {"all", "prepare", "accuracy", "status", "tests", "perf_auto", "perf_off",
               "perf_force", "summary"}
    stages = [item for item in args.stage.split(",") if item]
    unknown = [item for item in stages if item not in allowed]
    if unknown or not stages:
        print(f"unknown stages: {unknown}", file=sys.stderr)
        return 2
    jump = os.environ.get("A3_JUMP", "")
    password = os.environ.get("A3_PASSWORD", "")
    if not jump or not password:
        print("A3_JUMP and A3_PASSWORD must be set", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    project = args.project.resolve()
    if args.reuse_dir:
        remote_project = "/" + args.reuse_dir.lstrip("/")
        label = Path(remote_project).name
    else:
        remote_project = f"/workspace/{args.remote_prefix}_{stamp}"
        label = f"a3_round_{stamp}"
    remote_run = f"{remote_project}_run"
    results = project / "artifacts" / label
    results.mkdir(parents=True, exist_ok=True)
    stage_csv = ",".join(stages)

    client = SSH(jump, password)
    status = None
    try:
        with tempfile.TemporaryDirectory(prefix="a3-sync") as tmp:
            if args.reuse_dir:
                upload = ""
                stdin_handle = None
                print(f"reusing remote project: {remote_project}")
            else:
                archive_path = Path(tmp) / "source.tar.gz"
                make_archive(project, archive_path)
                print(f"source archive: {archive_path.stat().st_size} bytes")
                upload = f"mkdir -p '{remote_project}' && tar -xzf - -C '{remote_project}' && "
                stdin_handle = archive_path.open("rb")
            print(f"remote project: {remote_project}")
            print(f"remote run dir: {remote_run}")
            print(f"stages: {stage_csv}")
            remote = (
                f"{upload}cd '{remote_project}' && sed -i 's/\\r$//' tools/remote_round.sh && "
                f"bash tools/remote_round.sh '{remote_project}' '{args.task_dir}' '{remote_run}' '{stage_csv}'"
            )
            try:
                status = client.run(remote, stdin=stdin_handle, timeout=args.timeout)
            finally:
                if stdin_handle is not None:
                    stdin_handle.close()
            print(f"remote stage exited with {status.returncode}")
    finally:
        client.close()

    if status is None:
        print("remote stage did not start", file=sys.stderr)
        return 1
    if args.skip_download:
        return status.returncode
    try:
        client = SSH(jump, password)
        archive_path = results / "round_artifacts.tar.gz"
        with archive_path.open("wb") as handle:
            client.run(f"tar -czf - -C '{remote_run}' .", stdout=handle, timeout=900, check=True)
        client.close()
        with tarfile.open(archive_path) as extracted:
            extracted.extractall(results / "round", filter="data")
        print(f"results: {results}")
    except Exception as error:  # the token may already be expired; the log above is the result
        print(f"artifact download skipped: {error}", file=sys.stderr)
        print(f"fetch later with: tar -czf - -C '{remote_run}' .", file=sys.stderr)
    return status.returncode


if __name__ == "__main__":
    raise SystemExit(main())