"""Pre-rendered test images.

Every image the tests used to render with GalSim at run time lives in
``tests/data/<module>.fits``, one HDU per image, keyed by ``EXTNAME``.
``tests/data/make_fixtures.py`` is the only place that renders; the
parameters of each image are recorded in its header.
"""

import os

import anacal
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


def load_reference(name: str):
    """A single-HDU regression reference ``tests/data/<name>.fits``: an
    array AnaCal itself produced (see the README), not a rendered image."""
    return fitsio.read(os.path.join(DATA_DIR, name + ".fits"))


def symmetrized_spectrum(n, scale, variance, psf, klim, delta=0.05, corr=None):
    """Noise spectrum on an n x n grid, symmetrized with ``psf`` (the
    spectral renoising's patch step)."""

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


# The 670 x 670 COSMOS scene of fpfs_process_image.fits: 10 x 10 stamps of
# 64 pixels plus a 15-pixel border, one galaxy per stamp (g1 = +0.02, mag
# zero point 30), Moffat PSF.  The noisy end-to-end tests share it.
COSMOS_NSTAMP, COSMOS_NGRID, COSMOS_BUFF = 10, 64, 15
COSMOS_MAG_ZERO = 30.0


def cosmos_scene():
    """(galaxy image, PSF) of the COSMOS scene, both float64."""
    fix = load("fpfs_process_image")
    psf = np.asarray(fix["psf"], dtype=np.float64)
    gal = np.asarray(fix["gal"], dtype=np.float64)
    return gal, psf


def cosmos_centres():
    """The stamp centres of the COSMOS scene as a forced-measurement
    detection catalog (pixel 'y', 'x')."""
    c = np.arange(COSMOS_NSTAMP) * COSMOS_NGRID + COSMOS_NGRID // 2
    c = c + COSMOS_BUFF
    yy, xx = np.meshgrid(c, c, indexing="ij")
    det = np.zeros(c.size**2, dtype=[("y", np.float64), ("x", np.float64)])
    det["y"] = yy.ravel()
    det["x"] = xx.ravel()
    return det
