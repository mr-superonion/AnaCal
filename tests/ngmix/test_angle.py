import anacal
import numpy as np

from ..fixtures import load


def test_angle():
    scale = 0.2
    sigma_arcsec = 0.4
    # pre-rendered (tests/data/ngmix_angle.fits): a sheared Moffat PSF
    # (beta 2.5, fwhm 0.7) and an e1 = 0.2 exponential (hlr 0.16, flux
    # 150) rotated by `angle` degrees, both centred on pixel (32, 32)
    angle = 30.0
    fix = load("ngmix_angle")
    psf_array = fix["psf"]
    img_array = fix["gal"]

    num_epochs = 20
    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        stamp_size=32,
    )
    src = anacal.table.galNumber()
    src.model.x1.v = 32 * scale
    src.model.x2.v = 32 * scale
    src.model.F.v = 1.0
    src.model.t.v = -0.5
    catalog = [src]
    prior = anacal.ngmix.modelPrior()

    cat = fitter.process_cell(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]

    np.testing.assert_allclose(
        cat.model.t.v / np.pi * 180, angle,
        atol=0.0, rtol=1e-4,
    )
