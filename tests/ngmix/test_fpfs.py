"""FPFS moments inside the Gaussian fitter (ngmix.GaussFit): switching
them off.  Their values and responses are tested against the forced FPFS
measurement in tests/fpfs/test_task_force_consistency.py.
"""

import anacal
import numpy as np


def test_ngmix_fpfs_disabled():
    nx = 8
    ny = 8
    scale = 0.2
    sigma_shapelets = 0.3

    psf_array = np.ones((ny, nx), dtype=float)
    img_array = np.zeros((ny, nx), dtype=float)

    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_shapelets,
        stamp_size=8,
        do_fpfs=False,
    )

    src = anacal.table.galNumber()
    src.model.x1.v = (nx / 2) * scale
    src.model.x2.v = (ny / 2) * scale
    src.x1_det = src.model.x1.v
    src.x2_det = src.model.x2.v
    src.model.F.v = 1.0
    src.model.t.v = -0.5

    catalog = [src]
    prior = anacal.ngmix.modelPrior()

    cat = fitter.process_cell(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=1,
        variance=0.1,
    )[0]

    np.testing.assert_allclose(cat.fpfs_m0.v, 0.0)
    np.testing.assert_allclose(cat.fpfs_m2.v, 0.0)
