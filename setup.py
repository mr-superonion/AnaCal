import glob
import os

import pybind11
from setuptools import Extension, setup

conda_prefix = os.environ.get("CONDA_PREFIX")
include_dirs = ["include/"]
if conda_prefix:
    include_dirs.append(os.path.join(conda_prefix, "include"))

# In-tree extension builds (e.g. `python setup.py build_ext --inplace`)
# leave _anacal.cpython-*.so inside the package directory.  Those stale
# binaries would be packaged into the wheel as package data and silently
# OVERWRITE the freshly compiled extension, so remove them before every
# build.
for _stray in glob.glob("python/anacal/*.so"):
    print(f"removing stale in-tree extension: {_stray}")
    os.remove(_stray)


include_dirs.append(pybind11.get_include())


ext_modules = []
ext_modules.append(
    Extension(
        "anacal._anacal",  # Name of the module
        [
            "python/anacal/_anacalLib.cc",
            "src/image.cpp",
            "src/model.cpp",
            "src/fpfs.cpp",
            "src/fpfs/base.cpp",
            "src/fpfs/image.cpp",
            "src/fpfs/catalog.cpp",
            "src/fpfs/force.cpp",
            "src/noise.cpp",
            "src/mask.cpp",
            "src/math.cpp",
            "src/ngmix.cpp",
            "src/table.cpp",
            "src/detector.cpp",
            "src/geometry.cpp",
            "src/task.cpp",
            "src/psfmodel.cpp",
        ],
        include_dirs=include_dirs,
        # Almost all of the implementation lives in these headers; listing
        # them makes setuptools rebuild when a header changes, instead of
        # silently keeping a stale extension.
        depends=sorted(
            glob.glob("include/anacal/**/*.h", recursive=True)
        ),
        language="c++",
        extra_compile_args=[
            "-Wall",
            "-Wextra",
            "-Wdeprecated-declarations",
            "-std=c++17",
            "-O3",
            "-fvisibility=hidden",
        ],
        extra_link_args=["-flto"],
    )
)

setup(ext_modules=ext_modules)
