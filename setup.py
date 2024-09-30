import ctypes
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
        lib_ext = ".dylib"
    else:
        lib_ext = ".so"
    name = "libfftw3" + lib_ext
    tried_dirs = set()  # Keep track, so we don't try the same thing twice.
    for dir in try_libdirs:
        if dir == "":
            continue  # This messes things up if it's in there.
        if dir in tried_dirs:
            continue
        else:
            tried_dirs.add(dir)
        if not os.path.isdir(dir):
            continue
        libpath = os.path.join(dir, name)
        if not os.path.isfile(libpath):
            continue
        try:
            lib = ctypes.cdll.LoadLibrary(libpath)
            return libpath
        except OSError as e:
            # Some places use lib64 rather than/in addition to lib.
            if dir.endswith("lib") and os.path.isdir(dir + "64"):
                dir += "64"
                try:
                    libpath = os.path.join(dir, name)
                    if not os.path.isfile(libpath):
                        continue
                    lib = ctypes.cdll.LoadLibrary(libpath)
                    return libpath
                except OSError:
                    pass

    # If we didn't find it anywhere, but the user has set FFTW_DIR, trust it.
    if "FFTW_DIR" in os.environ:
        libpath = os.path.join(os.environ["FFTW_DIR"], name)
        print("WARNING:")
        print("Could not find an installed fftw3 library named %s" % name)
        print(
            "Trust the provided FFTW_DIR=%s for the library location." % libpath
        )
        print("If this is incorrect, you may have errors later when linking.")
        return libpath


fftw_lib = find_fftw_lib()
if fftw_lib is not None:
    fftw_libpath, fftw_libname = os.path.split(fftw_lib)
    include_dirs.append(
        os.path.join(
            os.path.split(fftw_libpath)[0],
            "include",
        )
    )

ext_modules = []
ext_modules.append(
    Extension(
        "anacal._anacal",  # Name of the module
        [
            "python/anacal/_anacalLib.cc",
            "src/image.cpp",
            "src/model.cpp",
            "src/fpfs.cpp",
            "src/fpfs/image.cpp",
            "src/fpfs/catalog.cpp",
            "src/noise.cpp",
            "src/psf.cpp",
            "src/mask.cpp",
        ],
        include_dirs=include_dirs,
        libraries=["fftw3"],
        language="c++",
        extra_compile_args=[
            "-Wall",
            "-Wextra",
            "-Wdeprecated-declarations",
            "-std=c++17",
            "-fopenmp",
            "-O3",
            "-fvisibility=hidden",
        ],
        extra_link_args=["-flto", "-fopenmp"],
    )
)

this_dir = os.path.dirname(os.path.realpath(__file__))
__version__ = ""
fname = os.path.join(this_dir, "python/anacal", "__version__.py")
with open(fname, "r") as ff:
    exec(ff.read())
long_description = open(os.path.join(this_dir, "README.md")).read()

setup(
    name="anacal",
    version=__version__,
    author="Xiangchong Li",
    author_email="mr.superonion@hotmail.com",
    python_requires=">=3.10",
    setup_requires=["pybind11>=2.2", "setuptools>=38", "wheel"],
    install_requires=[
        "pybind11>=2.2",
        "numpy",
        "galsim",
        "fitsio",
        "pydantic",
    ],
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
