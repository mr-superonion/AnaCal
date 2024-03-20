import os
from setuptools import setup, find_packages, Extension

ext_modules = []
ext_modules.append(
    Extension(
        "anacal.model",  # Name of the module
        ["src/model.cpp"],  # Source files
        include_dirs=["include/"],  # Include directories for header files
        language="c++",
        extra_compile_args=["-std=c++11", "-fopenmp", "-O3"],
        extra_link_args=["-flto", "-fopenmp"],
    )
)

this_dir = os.path.dirname(os.path.realpath(__file__))
__version__ = ""
fname = os.path.join(this_dir, "anacal", "__version__.py")
with open(fname, "r") as ff:
    exec(ff.read())
long_description = open(os.path.join(this_dir, "README.md")).read()

setup(
    name="anacal",
    version=__version__,
    author="Xiangchong Li",
    author_email="mr.superonion@hotmail.com",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "schwimmbad",
        "jax>=0.4.9",
        "jaxlib>=0.4.9",
        "galsim",
        "astropy",
        "matplotlib",
        "fitsio",
        "pybind11"
    ],
    packages=find_packages(),
    zip_safe=False,
    ext_modules=ext_modules,
    url="https://github.com/mr-superonion/AnaCal/",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
