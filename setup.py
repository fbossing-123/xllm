"""Build the target-only PyTorch NPU extension with CMake/Ascend C."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self):
        super().__init__("dual_affine_layernorm._dual_affine_layer_norm", sources=[])


class BuildExtension(build_ext):
    def build_extension(self, extension):
        root = Path(__file__).resolve().parent
        build_dir = root / "build" / "torch_extension"
        opp_root = Path(os.environ.get(
            "DUAL_AFFINE_LAYERNORM_OPP_ROOT",
            root / "build/packages/vendors/dual_affine_layernorm_custom"))
        build_dir.mkdir(parents=True, exist_ok=True)
        built = Path(self.get_ext_fullpath(extension.name))
        output_dir = built.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "cmake", "-S", str(root / "csrc"), "-B", str(build_dir),
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DTORCH_EXTENSION_OUTPUT_DIR={output_dir}",
            f"-DTORCH_EXTENSION_OUTPUT_NAME={built.name}",
            f"-DSOC_VERSION={os.environ.get('DUAL_AFFINE_LAYERNORM_SOC', 'ascend910_9382')}",
            f"-DOPP_ROOT={opp_root}",
        ], cwd=root, check=True)
        subprocess.run(["cmake", "--build", str(build_dir), "--parallel", str(os.environ.get("JOBS", "4"))],
                       cwd=root, check=True)
        kernel_library = build_dir / "lib" / "libdual_affine_layer_norm_torch_kernel.so"
        shutil.copy2(kernel_library, output_dir / kernel_library.name)
        opapi_library = opp_root / "op_api" / "lib" / "libcust_opapi.so"
        shutil.copy2(opapi_library, output_dir / opapi_library.name)
        if self.inplace:
            shutil.copy2(built, root / "dual_affine_layernorm" / built.name)

    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        kernel_name = "libdual_affine_layer_norm_torch_kernel.so"
        package_dir = Path(self.get_finalized_command("build_py").get_package_dir("dual_affine_layernorm"))
        for kernel_name in ["libdual_affine_layer_norm_torch_kernel.so", "libcust_opapi.so"]:
            source = Path(self.build_lib) / "dual_affine_layernorm" / kernel_name
            shutil.copy2(source, package_dir / kernel_name)


setup(
    name="dual-affine-layernorm",
    version="1.0.0",
    packages=["dual_affine_layernorm"],
    package_data={"dual_affine_layernorm": ["*.so"]},
    ext_modules=[CMakeExtension()],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
