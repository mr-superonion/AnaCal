import ctypes
import glob
import os
import sys

import pybind11
from setuptools import Extension, find_packages, setup

conda_prefix = os.environ.get("CONDA_PREFIX")
include_dirs = ["include/"]
if conda_prefix:
    include_dirs.append(os.path.join(conda_prefix, "include"))


include_dirs.append(pybind11.get_include())


def find_fftw_lib():
    import distutils.sysconfig

    try_libdirs = []

    # Start with the explicit FFTW_DIR, if present.
    if "FFTW_DIR" in os.environ:
        try_libdirs.append(os.environ["FFTW_DIR"])
        try_libdirs.append(os.path.join(os.environ["FFTW_DIR"], "lib"))
        try_libdirs.append(os.path.join(os.environ["FFTW_DIR"], "bin"))

    # Add the python system library directory.
    try_libdirs.append(distutils.sysconfig.get_config_var("LIBDIR"))

    # Try some standard locations where things get installed
    try_libdirs.extend(["/usr/local/lib", "/usr/lib"])
    if sys.platform == "darwin":
        try_libdirs.extend(["/sw/lib", "/opt/local/lib"])

    # Check the directories in LD_LIBRARY_PATH.
    # This doesn't work on OSX >= 10.11
    for path in ["LIBRARY_PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"]:
        if path in os.environ:
            for dir in os.environ[path].split(":"):
                try_libdirs.append(dir)

    # The user's home directory is often a good place to check.
    try_libdirs.append(os.path.join(os.path.expanduser("~"), "lib"))
    if sys.platform == "darwin":
        lib_names = ["libfftw3.dylib"]
    elif sys.platform == "win32":
        lib_names = [
            "libfftw3-3.dll",
            "fftw3-3.dll",
            "libfftw3.dll",
            "fftw3.dll",
        ]
    else:
        lib_names = ["libfftw3.so"]
    tried_dirs = set()  # Keep track, so we don't try the same thing twice.
    for dir in try_libdirs:
        if not dir:
            continue  # This messes things up if it's in there.
        if dir in tried_dirs:
            continue
        else:
            tried_dirs.add(dir)
        if not os.path.isdir(dir):
            continue
        for name in lib_names:
            libpath = os.path.join(dir, name)
            if not os.path.isfile(libpath):
                continue
            try:
                ctypes.cdll.LoadLibrary(libpath)
                return libpath
            except OSError:
                # Some places use lib64 rather than/in addition to lib.
                if dir.endswith("lib") and os.path.isdir(dir + "64"):
                    dir64 = dir + "64"
                    libpath = os.path.join(dir64, name)
                    if not os.path.isfile(libpath):
                        continue
                    try:
                        ctypes.cdll.LoadLibrary(libpath)
                        return libpath
                    except OSError:
                        pass

    # If we didn't find it anywhere, but the user has set FFTW_DIR, trust it.
    if "FFTW_DIR" in os.environ:
        # Try to locate any of the expected names inside FFTW_DIR
        for name in lib_names:
            libpath = os.path.join(os.environ["FFTW_DIR"], name)
            if os.path.isfile(libpath):
                return libpath
        print("WARNING:")
        print("Could not find an installed fftw3 library.")
        print(
            "Trust the provided FFTW_DIR=%s for the library location." % os.environ["FFTW_DIR"]
        )
        print("If this is incorrect, you may have errors later when linking.")
        return os.environ["FFTW_DIR"]


fftw_lib = find_fftw_lib()
library_dirs = []
libraries = ["fftw3"]
if fftw_lib is not None:
    if os.path.isfile(fftw_lib):
        fftw_libpath, fftw_libname = os.path.split(fftw_lib)
        include_roots = {os.path.split(fftw_libpath)[0], fftw_libpath}
    else:
        fftw_libpath = fftw_lib
        fftw_libname = ""
        include_roots = {fftw_libpath, os.path.split(fftw_libpath)[0]}

    for include_root in include_roots:
        candidate_include = os.path.join(include_root, "include")
        if os.path.isdir(candidate_include) and candidate_include not in include_dirs:
            include_dirs.append(candidate_include)

    if os.path.isdir(fftw_libpath) and fftw_libpath not in library_dirs:
        library_dirs.append(fftw_libpath)

    if sys.platform == "win32":
        lib_dir = os.path.join(os.path.split(fftw_libpath)[0], "lib")
        if os.path.isdir(lib_dir):
            if lib_dir not in library_dirs:
                library_dirs.append(lib_dir)
            lib_candidates = glob.glob(os.path.join(lib_dir, "*fftw*.lib"))
            if lib_candidates:
                libraries = [os.path.splitext(os.path.basename(lib_candidates[0]))[0]]
            elif fftw_libname:
                libraries = [os.path.splitext(fftw_libname)[0]]
        elif fftw_libname:
            libraries = [os.path.splitext(fftw_libname)[0]]

extra_compile_args = [
    "-Wall",
    "-Wextra",
    "-Wdeprecated-declarations",
    "-std=c++17",
    "-fopenmp",
    "-O3",
    "-fvisibility=hidden",
]
extra_link_args = ["-flto", "-fopenmp"]

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2", "/openmp", "/EHsc"]
    extra_link_args = ["/openmp"]

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
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
)

this_dir = os.path.dirname(os.path.realpath(__file__))
long_description = open(os.path.join(this_dir, "README.md")).read()

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
    url="https://github.com/mr-superonion/AnaCal/",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
