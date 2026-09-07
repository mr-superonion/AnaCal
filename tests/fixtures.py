"""Pre-rendered test images.

Every image the tests used to render with GalSim at run time lives in
``tests/data/<module>.fits``, one HDU per image, keyed by ``EXTNAME``.
``tests/data/make_fixtures.py`` is the only place that renders; the
parameters of each image are recorded in its header.
"""

import os

import fitsio

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
