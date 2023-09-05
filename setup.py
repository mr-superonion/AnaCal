import os
from setuptools import setup, find_namespace_packages

__version__ = ""
fname = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "anacal", "__version__.py"
)
with open(fname, "r") as ff:
    exec(ff.read())

include_modules = ["fpfs", "impt"]
include_packages = ["anacal.%s.%s" %(sub, sub) for sub in include_modules]
setup(
    name="anacal",
    version=__version__,
    packages=find_namespace_packages(
        include=include_packages,
    ),
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
    ],
)
