"""High-level interface to AnaCal, the astronomical calibration library.

Exposes the compiled :mod:`_anacal` extension alongside convenient
submodules such as :mod:`fpfs`, :mod:`psf` (the PSF interface and the
native coadd-PSF models), and general utilities.
"""

from . import _anacal, fpfs, psf, utils
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
