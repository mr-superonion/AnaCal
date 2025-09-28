import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    include_paths as torch_include_paths,
    library_paths as torch_library_paths,
)


def _unique(items):
    seen = set()
    unique_items = []
    for item in items:
        if item and item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


conda_prefix = os.environ.get("CONDA_PREFIX")
include_dirs = ["include/"]
library_dirs = []

if conda_prefix:
    include_dirs.append(os.path.join(conda_prefix, "include"))
    library_dirs.append(os.path.join(conda_prefix, "lib"))

torch_home = os.environ.get("TORCH_HOME")
if torch_home:
    include_dirs.append(os.path.join(torch_home, "include"))
    library_dirs.append(os.path.join(torch_home, "lib"))

include_dirs.extend(torch_include_paths())
library_dirs.extend(torch_library_paths())

include_dirs = _unique(include_dirs)
library_dirs = _unique(library_dirs)

ext_modules = [
    CppExtension(
        "anacal._anacal",
        [
            "python/anacal/_anacalLib.cc",
            "src/image.cpp",
            "src/model.cpp",
            "src/fpfs.cpp",
            "src/fpfs/base.cpp",
            "src/fpfs/image.cpp",
            "src/fpfs/catalog.cpp",
            "src/noise.cpp",
            "src/mask.cpp",
            "src/math.cpp",
            "src/ngmix.cpp",
            "src/table.cpp",
            "src/detector.cpp",
            "src/geometry.cpp",
            "src/task.cpp",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        define_macros=[("USE_TORCH_FFT", None)],
        extra_compile_args={
            "cxx": [
                "-Wall",
                "-Wextra",
                "-Wdeprecated-declarations",
                "-std=c++17",
                "-fopenmp",
                "-O3",
                "-fvisibility=hidden",
            ]
        },
        extra_link_args=["-flto", "-fopenmp"],
    )
]

this_dir = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="anacal",
    author="Xiangchong Li",
    author_email="mr.superonion@hotmail.com",
    python_requires=">=3.10",
    install_requires=[
        "pybind11>=2.2",
        "numpy",
        "galsim",
        "fitsio",
        "pydantic",
        "torch>=2.0",
    ],
    use_scm_version=True,
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    include_package_data=True,
    zip_safe=False,
    package_data={
        "anacal": ["data/*.fits"],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    url="https://github.com/mr-superonion/AnaCal/",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
