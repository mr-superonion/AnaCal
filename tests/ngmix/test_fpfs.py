import anacal
import numpy as np

from ..fixtures import load


def test_ngmix_fpfs():
    nx = 64
    ny = 64
    scale = 0.2
    sigma_shapelets = 0.38
    # pre-rendered (tests/data/ngmix_fpfs.fits): a sheared Moffat PSF
    # (beta 2.5, fwhm 0.8) and an e1 = 0.2 exponential (hlr 0.21, flux
    # 150, rotated 30 deg) centred on pixel (32, 32)
    fix = load("ngmix_fpfs")
    psf_array = fix["psf"]
    img_array = fix["gal"]

    num_epochs = 20
    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_shapelets,
        stamp_size=32,
        fpfs_c0=0.0,
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
    m00 = cat.fpfs_m0.v
    trace = cat.fpfs_m2.v
    m22c = cat.fpfs_e1.v * cat.fpfs_m0.v
    m22s = cat.fpfs_e2.v * cat.fpfs_m0.v

    sigma_shapelets_fpfs = sigma_shapelets * np.sqrt(2.0)

    ftask = anacal.fpfs.FpfsTask(
        npix=64,
        pixel_scale=scale,
        sigma_shapelets=sigma_shapelets_fpfs,
        psf_array=psf_array,
    )

    coords = np.array(
        [(ny / 2.0 , nx / 2.0)],
        dtype=[
            ("y", "f8"),
            ("x", "f8"),
        ],
    )

    src = ftask.run(
        gal_array=img_array,
        psf=psf_array,
        det=coords,
    )["data"]

    np.testing.assert_allclose(
        src["m00"][0], m00,
        atol=0.0, rtol=1e-4,
    )

    np.testing.assert_allclose(
        src["m00"][0] + src["m20"][0], trace,
        atol=0.0, rtol=1e-4,
    )

    np.testing.assert_allclose(
        src["m22c"][0], m22c,
        atol=0.0, rtol=1e-4,
    )

    np.testing.assert_allclose(
        src["m22s"][0], m22s,
        atol=0.0, rtol=1e-4,
    )


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
