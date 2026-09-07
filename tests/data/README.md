# Pre-rendered test images

The tests do not render images at run time and AnaCal does not depend on
GalSim, not even for testing: the GalSim Lanczos kernels the PIFF path
must reproduce are stored too (`psf_lanczos.fits`).  Every image a test needs is stored here, one multi-extension
FITS per test module (`fpfs_*.fits`, `ngmix_*.fits`, `task_*.fits`,
`image*.fits`), one HDU per image keyed by `EXTNAME`, with the
parameters that produced it in the HDU header.  `tests/fixtures.py`
reads them.

`make_fixtures.py` is the only renderer.  It needs GalSim and
`cosmos_sim.py` (formerly `anacal.simulation`, a COSMOS-catalogue
galaxy simulator) with `src_cosmos.fits`, the GalSim COSMOS 25.2
magnitude-limited sample (see
https://galsim-developers.github.io/GalSim/_build/html/real_gal.html).
Neither is collected by pytest nor installed with the package.  Re-run
it only when an image genuinely has to change: several tests compare
against numbers derived from these exact pixels.

    cd tests/data && python make_fixtures.py

## Checking that nothing needs GalSim

Shadow the package with an empty module on `PYTHONPATH`; this reaches
the notebook kernels nbval starts as well as pytest itself:

    mkdir -p /tmp/nogalsim && echo 'raise ImportError("galsim is not installed")' > /tmp/nogalsim/galsim.py
    PYTHONPATH=/tmp/nogalsim pytest

Nothing may skip.
