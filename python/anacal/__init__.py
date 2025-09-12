"""High-level interface to AnaCal, the astronomical calibration library.

Exposes the compiled :mod:`_anacal` extension alongside convenient
submodules such as :mod:`fpfs`, :mod:`psf`, :mod:`simulation`, and general
utilities.
"""

from . import _anacal, base, fpfs, simulation, utils, psf
from .__version__ import __version__  # noqa
from ._anacal import (
    detector,
    geometry,
    image,
    mask,
    math,
    model,
    ngmix,
    noise,
    table,
    task,
)

__all__ = [
    "_anacal",
    "image",
    "fpfs",
    "psf",
    "model",
    "base",
    "noise",
    "mask",
    "simulation",
    "utils",
    "math",
    "ngmix",
    "detector",
    "geometry",
    "table",
    "task",
]
