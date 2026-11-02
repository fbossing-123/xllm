"""SHA-256 fingerprints for source, built artifacts and generated reports."""

import hashlib
import json
from pathlib import Path
import platform
import sys


SOURCE_ROOTS = (
    "dual_affine_layernorm", "op_host", "op_kernel", "kernel_direct", "csrc",
    "examples", "tests", "tools", "docs",
)
ROOT_FILES = (
    "CMakeLists.txt", "setup.py", "build.sh", "install_private.sh", "solution.py", "README.md",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def delivery_files(project_root, include_reports=False):
    root = Path(project_root).resolve()
    files = set()
    for name in SOURCE_ROOTS:
        directory = root / name
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    files.add(path)
    for name in ROOT_FILES:
        path = root / name
        if path.is_file():
            files.add(path)
    for pattern in ("*.run", "libcust_opapi.so", "libdual_affine_layer_norm*.so"):
        build = root / "build"
        if build.is_dir():
            files.update(path for path in build.rglob(pattern) if path.is_file())
    package = root / "dual_affine_layernorm"
    if package.is_dir():
        files.update(path for path in package.glob("*.so") if path.is_file())
    if include_reports:
        submission = root / "task_submission"
        files.update(path for path in submission.iterdir() if path.is_file())
    return sorted(files)


def create_manifest(project_root, include_reports=False):
    root = Path(project_root).resolve()
    files = {}
    for path in delivery_files(root, include_reports):
        relative = path.relative_to(root).as_posix()
        category = "build_artifact" if relative.startswith("build/") or path.suffix == ".so" else "source"
        if relative.startswith("task_submission/"):
            category = "report"
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size, "category": category}
    return {
        "schema_version": 1,
        "project_root_name": root.name,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "files": files,
    }


def write_manifest(project_root, output, include_reports=False):
    manifest = create_manifest(project_root, include_reports)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_implementation_fingerprints(run_dir, project_root):
    """Bind formal evidence to the exact source and binaries being delivered."""
    import json
    path = Path(run_dir) / "raw" / "implementation_fingerprints.json"
    if not path.is_file():
        raise ValueError("implementation fingerprint evidence is missing")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = create_manifest(project_root, include_reports=False)
    if recorded.get("files") != current["files"]:
        raise ValueError("implementation files changed or mismatch the acceptance evidence")
    loaded = recorded.get("loaded_binaries")
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("implementation loaded-binary evidence is missing")
    current_binaries = {}
    for relative, metadata in current["files"].items():
        if metadata["category"] == "build_artifact":
            current_binaries.setdefault(Path(relative).name, set()).add(metadata["sha256"])
    for binary in loaded:
        name = Path(binary.get("path", "")).name
        if binary.get("sha256") not in current_binaries.get(name, set()):
            raise ValueError(f"implementation loaded binary is not delivered unchanged: {name}")
    names = {Path(binary.get("path", "")).name for binary in loaded}
    required = ("libdual_affine_layer_norm_torch_kernel.so", "libcust_opapi.so")
    if any(name not in names for name in required) or not any(
            name.startswith("_dual_affine_layer_norm.") for name in names):
        raise ValueError("implementation loaded-binary evidence is incomplete")
    return recorded
