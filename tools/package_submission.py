#!/usr/bin/env python3
"""Package development sources or a formally verified submission, never confuse them."""

import argparse
import hashlib
from pathlib import Path
import zipfile

try:
    from .build_reports import REQUIRED_REPORTS, validate_evidence
    from .fingerprint import create_manifest, delivery_files, sha256_file
except ImportError:
    from build_reports import REQUIRED_REPORTS, validate_evidence
    from fingerprint import create_manifest, delivery_files, sha256_file


def package_submission(project_root, zip_path, run_dir=None, development=False):
    root = Path(project_root).resolve()
    output = Path(zip_path).resolve()
    development = bool(development)
    if development and output.name == "task_submission.zip":
        raise ValueError("development archive cannot be named task_submission.zip")
    if not development and output.name != "task_submission.zip":
        raise ValueError("formal submission archive must be named task_submission.zip")
    if output.exists():
        raise ValueError(f"refusing to overwrite existing archive: {output}")
    if development:
        if "development" not in output.name and "unverified" not in output.name:
            raise ValueError("development archive name must contain development or unverified")
        manifest = create_manifest(root, include_reports=False)
    else:
        if run_dir is None:
            raise ValueError("formal packaging requires --run-dir with complete passing A3 evidence")
        evidence = validate_evidence(run_dir, project_root=root)
        submission = root / "task_submission"
        missing = [name for name in REQUIRED_REPORTS if not (submission / name).is_file()]
        if missing:
            raise ValueError(f"formal reports are missing: {missing}")
        manifest = create_manifest(root, include_reports=True)
        manifest["verification"] = {
            "status": "formal_A3_passed",
            "run_summary": str((Path(run_dir).resolve() / "summary.json")),
            "environment": evidence["environment"],
            "totals": {"public": 46, "extra": 24, "invalid": 60, "performance": 10,
                       "memory_records": 20, "cross_stream_records": 2,
                       "verifier_counterexamples": len(evidence["verifier"])},
        }
        (submission / "manifest.json").write_text(
            __import__("json").dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files = delivery_files(root, include_reports=not development)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            if relative == "task_submission/manifest.json":
                continue
            archive.write(path, relative)
        if not development:
            archive.writestr("task_submission/manifest.json",
                             __import__("json").dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        names = set(archive.namelist())
        for relative, metadata in manifest["files"].items():
            if relative == "task_submission/manifest.json" or relative not in names:
                continue
            digest = hashlib.sha256(archive.read(relative)).hexdigest()
            if digest != metadata["sha256"]:
                raise ValueError(f"ZIP hash mismatch: {relative}")
        if not development:
            for name in REQUIRED_REPORTS:
                if f"task_submission/{name}" not in names:
                    raise ValueError(f"ZIP is missing required report: {name}")
    return {"archive": str(output), "files": len(files), "sha256": sha256_file(output),
            "status": "development_unverified" if development else "formal_A3_passed"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args(argv)
    result = package_submission(args.project_root, args.output, args.run_dir, args.development)
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
