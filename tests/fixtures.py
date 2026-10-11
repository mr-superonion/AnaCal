"""Pre-rendered test images.

Every image the tests used to render with GalSim at run time lives in
``tests/data/<module>.fits``, one HDU per image, keyed by ``EXTNAME``.
``tests/data/make_fixtures.py`` is the only place that renders; the
parameters of each image are recorded in its header.
"""

import os

import fitsio
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load(name: str) -> dict:
    """All image HDUs of ``tests/data/<name>.fits`` as ``{EXTNAME: array}``."""
    fname = os.path.join(DATA_DIR, name + ".fits")
    out = {}
    with fitsio.FITS(fname) as ff:
        for hdu in ff:
            if hdu.has_data():
                out[hdu.get_extname()] = hdu.read()
    return out


def symmetrized_spectrum(n, scale, variance, psf, klim, delta=0.05, corr=None):
    """Noise spectrum on an n x n grid, symmetrized with ``psf`` (the
    spectral renoising's patch step)."""
    import anacal

    ns = anacal.renoise.make_noise_spectrum(n, scale, variance, corr)
    return anacal.renoise.symmetrize_spectrum(ns, psf, klim, delta)


def make_filler(eps2, scale, variance, psf, klim, delta=0.05, corr=None):
    """The spin-2 filler g * eps2 of an image whose noise realization is
    ``eps2`` (any shape): g from the symmetrized spectrum on the smallest
    even square grid holding the image, applied by FFT and cropped."""
    ny, nx = eps2.shape
    n = max(ny, nx)
    n += n % 2
    sym = symmetrized_spectrum(n, scale, variance, psf, klim, delta, corr)
    pad = np.zeros((n, n))
    pad[:ny, :nx] = eps2
    out = np.fft.irfft2(np.fft.rfft2(pad) * sym.g, s=(n, n))
    return out[:ny, :nx]
