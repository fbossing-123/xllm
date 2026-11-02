#!/usr/bin/env python3
"""Persistent A3 shell over one jump-hosted SSH session.

start:  spawn a background daemon that keeps ``bash -s`` alive and replays any
        line appended to ``cmd.txt`` into its stdin.
send:   append one command line (no embedded newline) to the command file.
tail:   print the received output.
stop:   close the session.

Secrets come from A3_JUMP / A3_PASSWORD and are never written anywhere else.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parent.parent / "artifacts" / "a3_session"
CMD_FILE = BASE / "cmd.txt"
OUT_FILE = BASE / "out.log"
PID_FILE = BASE / "pid.txt"
READY_FILE = BASE / "ready.txt"


def build_argv(target: str):
    jump = os.environ["A3_JUMP"]
    password = os.environ["A3_PASSWORD"]
    directory = Path(tempfile.mkdtemp(prefix="a3-session"))
    password_file = directory / "p"
    password_file.write_text(password + "\n", encoding="utf-8")
    os.chmod(password_file, 0o600)
    askpass = directory / "askpass.sh"
    askpass.write_text("#!/bin/sh\ncat '%s'\n" % str(password_file).replace("'", "'\\''"),
                       encoding="utf-8")
    os.chmod(askpass, 0o700)
    env = dict(os.environ)
    env.update({"SSH_ASKPASS": str(askpass), "SSH_ASKPASS_REQUIRE": "force", "DISPLAY": ":0",
                "MSYS_NO_PATHCONV": "1", "MSYS2_ARG_CONV_EXCL": "*"})
    argv = [
        shutil.which("ssh"), "-T",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=240", "-o", "TCPKeepAlive=yes",
        "-o", "ConnectTimeout=30", "-o", "NumberOfPasswordPrompts=1",
        "-o", "PreferredAuthentications=password,keyboard-interactive", "-o", "PubkeyAuthentication=no",
        "-J", jump, target, "bash --noprofile --norc -s",
    ]
    return argv, env


def daemon(target: str) -> int:
    argv, env = build_argv(target)
    BASE.mkdir(parents=True, exist_ok=True)
    position = 0
    if CMD_FILE.exists():
        position = CMD_FILE.stat().st_size
    with OUT_FILE.open("ab") as output:
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE, stdout=output,
                                stderr=subprocess.STDOUT, bufsize=0)
        READY_FILE.write_text(str(proc.pid), encoding="utf-8")
        try:
            while proc.poll() is None:
                if CMD_FILE.exists():
                    data = CMD_FILE.read_bytes()
                    if len(data) > position:
                        chunk = data[position:]
                        position = len(data)
                        proc.stdin.write(chunk)
                        proc.stdin.flush()
                time.sleep(0.15)
        except Exception:
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass
            proc.wait(timeout=10)
            output.write(f"\n=== session exited {proc.returncode} ===\n".encode())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help=argparse.SUPPRESS, choices=["start", "send", "tail", "stop", "status", "daemon"])
    parser.add_argument("command", nargs="?")
    parser.add_argument("--target", default=os.environ.get("A3_TARGET", "root@199.103.53.243"))
    parser.add_argument("--lines", type=int, default=60)
    args = parser.parse_args()

    if args.action == "start":
        BASE.mkdir(parents=True, exist_ok=True)
        if CMD_FILE.exists():
            CMD_FILE.unlink()
        if OUT_FILE.exists():
            OUT_FILE.unlink()
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "daemon", "--target", args.target],
                         creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
                         close_fds=True)
        for _ in range(100):
            if READY_FILE.exists():
                print("daemon started")
                return 0
            time.sleep(0.2)
        print("daemon did not start", file=sys.stderr)
        return 1
    if args.action == "daemon":
        return daemon(args.target)
    if args.action == "send":
        if args.command is None:
            print("nothing to send", file=sys.stderr)
            return 2
        with CMD_FILE.open("a", encoding="utf-8", newline="") as handle:
            handle.write(args.command + "\n")
        return 0
    if args.action == "tail":
        if not OUT_FILE.exists():
            print("no output yet")
            return 0
        data = OUT_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in data[-args.lines:]:
            print(line)
        return 0
    if args.action == "status":
        if READY_FILE.exists():
            print("ready pid", READY_FILE.read_text(encoding="utf-8").strip())
        print("output bytes", OUT_FILE.stat().st_size if OUT_FILE.exists() else 0)
        return 0
    if args.action == "stop":
        if READY_FILE.exists():
            pid = int(READY_FILE.read_text(encoding="utf-8").strip())
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())