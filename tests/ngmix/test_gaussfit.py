import anacal
import numpy as np
import pytest

from ..fixtures import load


def test_ngmix_gaussian_fit_additive(test_g1=True):
    scale = 0.2
    sigma_arcsec = 0.4
    # pre-rendered (tests/data/ngmix_gaussfit.fits): a sheared Moffat
    # PSF (beta 2.5, fwhm 0.7) and a round exponential (hlr 0.2, flux
    # 150) offset by (-0.2, 0.11) pixels from the stamp centre
    fix = load("ngmix_gaussfit")
    psf_array = fix["psf"]

    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        stamp_size=32,
    )

    num_epochs = 35
    src = anacal.table.galNumber()
    src.model.x1.v = 32 * scale
    src.model.x2.v = 32 * scale
    src.x1_det = 32 * scale
    src.x2_det = 32 * scale
    src.model.F.v = 1.0
    src.model.t.v = -0.5
    catalog = [src]
    prior = anacal.ngmix.modelPrior()

    img_array = fix["gal_add"]
    cat_1 = fitter.process_cell(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]
    ell1 = cat_1.model.get_shape()[0]
    assert np.abs(ell1.v / ell1.g1) < 1e-4
    return


def test_ngmix_gaussian_fit2():
    nx = 64
    ny = 64
    scale = 0.2
    sigma_arcsec = 0.4
    dx1 = -0.2
    dx2 = 0.11
    # pre-rendered (tests/data/ngmix_gaussfit.fits): the same PSF and an
    # e = (0.2, -0.1) exponential (hlr 0.2) offset by (dx1, dx2) pixels
    # from the stamp centre, under the shears / fluxes / rotations used
    # below (g1p = +0.02, g1m = -0.02, f = flux, a = rotation in deg)
    fix = load("ngmix_gaussfit")
    psf_array = fix["psf"]

    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        stamp_size=32,
    )

    flux = 150.0
    num_epochs = 35
    src = anacal.table.galNumber()
    src.model.x1.v = nx // 2 * scale
    src.model.x2.v = nx // 2 * scale
    src.x1_det = nx // 2 * scale
    src.x2_det = nx // 2 * scale
    src.model.F.v = 1.0
    catalog = [src]
    prior = anacal.ngmix.modelPrior()

    # Test shear response calculation (no multiplicative bias)
    img_array = fix["gal_g1p_f150"]
    cat_1 = fitter.process_cell(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]
    ell1 = cat_1.model.get_shape()[0]

    assert np.abs((cat_1.model.F.v - flux) / flux) < 0.01
    np.testing.assert_allclose(
        cat_1.model.x1.v / scale - nx // 2, dx1,
        atol=1e-5, rtol=0,
    )
    np.testing.assert_allclose(
        cat_1.model.x2.v / scale - ny // 2, dx2,
        atol=1e-5, rtol=0,
    )
    np.testing.assert_allclose(
        cat_1.model.get_flux_stamp(128, 128, 0.1, 0.6).v,
        flux,
        atol=0,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        cat_1.model.get_flux_stamp(64, 64, 0.2, sigma_arcsec).v,
        flux,
        atol=0,
        rtol=1e-2,
    )

    img_array = fix["gal_g1m_f300"]
    cat_2 = fitter.process_cell(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=1.0,
    )[0]
    ell2 = cat_2.model.get_shape()[0]

    assert np.abs(
        (ell1.v - ell2.v) / (ell2.g1 + ell1.g1) * 2.0 / 0.04 - 1
    ) < 2e-3

    klim = 100.0
    img_obj = anacal.image.ImageQ(
        nx=nx,
        ny=ny,
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        klim=klim,
    )

    img_array1 = img_obj.prepare_qnumber_image(
        img_array,
        psf_array,
        xcen=img_array.shape[1] // 2,
        ycen=img_array.shape[0] // 2,
    )
    img_array2 = cat_2.model.get_image_stamp(nx, ny, scale, sigma_arcsec)
    diff = img_array1[0:3] - img_array2
    assert np.sum(np.abs(diff[0])) / np.sum(np.abs(img_array1[0])) < 2e-2
    assert np.sum(np.abs(diff[1])) / np.sum(np.abs(img_array1[1])) < 2e-1
    assert np.sum(np.abs(diff[2])) / np.sum(np.abs(img_array1[2])) < 2e-1

    # Test symmetry
    cat_1 = fitter.process_cell(
        catalog=catalog,
        img_array=fix["gal_g0_a0"],
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]
    ell1 = cat_1.model.get_shape()[0]

    cat_2 = fitter.process_cell(
        catalog=catalog,
        img_array=fix["gal_g0_a90"],
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]
    ell2 = cat_2.model.get_shape()[0]
    assert np.abs((ell2.v + ell1.v) / (ell2.g1 + ell1.g1)) < 5e-5

    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        stamp_size=32,
        force_size=True,
    )

    img_array = fix["gal_g1p_f150"]
    cat_1 = fitter.process_cell(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]

    ell1 = cat_1.model.get_shape()[0]
    assert ell1.v == 0
    assert ell1.g1 == 0
    assert ell1.g2 == 0

    np.testing.assert_allclose(
        cat_1.model.x1.v / scale - nx // 2, dx1,
        atol=1e-6, rtol=0.0,
    )
    np.testing.assert_allclose(
        cat_1.model.x2.v / scale - ny // 2, dx2,
        atol=1e-6, rtol=0.0,
    )
    img_array = fix["gal_g1m_f150"]
    cat_2 = fitter.process_cell(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=1.0,
    )[0]

    ell1 = cat_1.fpfs_e1
    ell2 = cat_2.fpfs_e1
    assert np.abs(
        (ell1.v - ell2.v) / (ell2.g1 + ell1.g1) * 2.0 / 0.04 - 1
    ) < 2e-3

    return


WIDE_CENTERS = [(31.2, 31.2), (95.9, 32.05), (160, 32.1), (224, 31.8)]
WIDE_FLUXES = [12, 23, 8.5, 18.4]


def _fit_wide(key, force_center, num_epochs=25, offset=0.0, **kw):
    """The four round galaxies sheared by g1 = 0.03 on the 256 x 64 strip
    (tests/data/ngmix_gaussfit.fits), started ``offset`` pixels
    (+x, -y) from their true centres."""
    fix = load("ngmix_gaussfit")
    fitter = anacal.ngmix.GaussFit(
        scale=0.2, sigma_arcsec=0.4, stamp_size=48,
        force_center=force_center, **kw
    )
    catalog = []
    for cx, cy in WIDE_CENTERS:
        src = anacal.table.galNumber()
        src.model.x1.v = (cx + offset) * 0.2
        src.model.x2.v = (cy - offset) * 0.2
        src.x1_det = src.model.x1.v
        src.x2_det = src.model.x2.v
        catalog.append(src)
    return fitter.process_cell(
        catalog=catalog,
        img_array=fix[key],
        psf_array=fix["psf_wide"],
        prior=anacal.ngmix.modelPrior(),
        num_epochs=num_epochs,
        variance=1.0,
    )


def _max_m(result):
    return max(
        abs(rr.model.get_shape()[0].v / rr.model.get_shape()[0].g1 / 0.03 - 1)
        for rr in result
    )


@pytest.mark.parametrize(
    "key, flux_rtol",
    [("gal_wide", 1e-2), ("gal_wide_exp", 5e-2), ("gal_wide_bd", 5e-2)],
)
def test_ngmix_gaussian_fit4(key, flux_rtol):
    """Shear recovery per galaxy, e1 / (de1/dg1) = g1, with the centre
    forced to the truth and the gate on at a relative tolerance of 1e-4
    (closes after two epochs), for the profile the model is (Gaussian)
    and two it is not (exponential, bulge + disk).  The response
    calibrates the misspecified model as well as the exact one; only the
    model flux differs from the total flux for the non-Gaussian
    profiles."""
    result = _fit_wide(key, force_center=True, conv_tol=1e-4)
    for rr, flux in zip(result, WIDE_FLUXES):
        assert rr.converged and rr.n_epochs < 5
        [e1, e2] = rr.model.get_shape()
        assert abs(e1.v / e1.g1 / 0.03 - 1.0) < 1e-3
        assert abs(e2.v / e2.g2) < 2e-5
        np.testing.assert_allclose(rr.model.F.v, flux, rtol=flux_rtol)


@pytest.mark.parametrize("key", ["gal_wide", "gal_wide_exp", "gal_wide_bd"])
def test_free_centre_off_truth_needs_small_tolerance_and_many_epochs(key):
    """A free centre started half a pixel from the truth.  The
    per-galaxy identity e/R = g is a shape-noise-free construction that
    holds only once the estimator has settled on the galaxy; an
    unconverged centroid adds a term to each galaxy's e that the four
    galaxies do not cancel, so the per-galaxy ratio scatters (scatter,
    not bias: the exported response is the exact derivative of the
    estimator each galaxy went through).  With the gate at a relative
    tolerance of 1e-10 and a cap of 50 it closes at 8-9 epochs with the
    centroid within 2e-6 pixel and |m| < 1e-3 for all three profiles;
    the default (gate off) reaches the same in a fixed 10 epochs.  A
    loose relative tolerance of 1e-3 closes at 4 epochs with the
    centroid 1e-2 pixel off and |m| of order one, and 5 fixed epochs
    leave |m| ~ 2e-2.  With the centre forced the same galaxies recover
    the shear to 1e-4 in two epochs at any tolerance
    (test_ngmix_gaussian_fit4)."""
    good = _fit_wide(key, False, num_epochs=50, offset=0.5, conv_tol=1e-10)
    assert _max_m(good) < 1e-3
    for rr, (cx, cy) in zip(good, WIDE_CENTERS):
        assert rr.converged and rr.n_epochs < 50
        assert abs(rr.model.x1.v / 0.2 - cx) < 1e-4
        assert abs(rr.model.x2.v / 0.2 - cy) < 1e-4
    assert _max_m(_fit_wide(key, False, num_epochs=10, offset=0.5)) < 1e-3
    assert _max_m(_fit_wide(key, False, 10, 0.5, conv_tol=1e-3)) > 1e-1
    assert _max_m(_fit_wide(key, False, 5, 0.5)) > 1e-3


def test_ngmix_gaussian_fit4_free_centre_response():
    """With a free centre the fitted centroid of a source at offset r
    from the shear centre responds to the shear as dx/dg = r (and to
    g2 with the axes swapped).  The centres start at the truth, so the
    value is converged from the first epoch and any gate would close
    before the centroid response has converged; the default, fixed
    epochs, converges it (to 2e-8 in 10 epochs with the relative
    damping).
    """
    nx, ny, scale = 256, 64, 0.2
    result = _fit_wide("gal_wide", force_center=False)
    for rr in result:
        np.testing.assert_allclose(
            rr.model.x1.g1 - (rr.model.x1.v - nx / 2 * scale),
            0.0, atol=1e-5, rtol=0,
        )
        np.testing.assert_allclose(
            rr.model.x1.g2 - (rr.model.x2.v - ny / 2 * scale),
            0.0, atol=1e-5, rtol=0,
        )
        np.testing.assert_allclose(
            rr.model.x2.g2 - (rr.model.x1.v - nx / 2 * scale),
            0.0, atol=1e-5, rtol=0,
        )
        np.testing.assert_allclose(
            rr.model.x2.g1 + (rr.model.x2.v - ny / 2 * scale),
            0.0, atol=1e-5, rtol=0,
        )
        [e1, e2] = rr.model.get_shape()
        assert abs(e1.v / e1.g1 / 0.03 - 1.0) < 1e-3


# # Loss function
# num_epochs=1
# loss_array = []
# ddloss_array = []
# t_array = np.arange(-5, 5.0, 0.005)
# for t in t_array:
#     src.model.t.v = t
#     # Test shear response calculation (no multiplicative bias)
#     cat_1 = fitter.process_cell(
#         catalog=catalog,
#         img_array=make_sim(g1=-0.02, g2=0.0, flux=flux),
#         psf_array=psf_array,
#         prior=prior,
#         num_epochs=num_epochs,
#         variance=1.0,
#     )[0]
#     loss_array.append(cat_1.loss.v.v)
#     ddloss_array.append(cat_1.loss.v_tt.v)
# loss_array = np.array(loss_array)
# ddloss_array = np.array(ddloss_array)

# src.model.t.v = -1.5
# plt.close()
# plt.plot(t_array, loss_array)
# plt.plot(t_array, ddloss_array)
# plt.xlabel("log(R)")
# plt.ylabel("loss")
# plt.axhline(0.0, ls='--')


# (sigma [arcsec], g1, g2, flux) and centres (pixels) of the four Gaussians
# in tests/data/ngmix_gausscov.fits; duplicated from make_fixtures.py.
GAUSSCOV_CASES = [
    (0.20, 0.0, 0.0, 20.0),
    (0.25, 0.15, 0.0, 30.0),
    (0.30, 0.0, -0.2, 40.0),
    (0.35, -0.1, 0.12, 50.0),
]
GAUSSCOV_CENTERS = [(31.2, 31.7), (95.6, 32.3), (160.4, 31.9), (223.8, 32.1)]


def _gaussian_covariance(sigma, g1, g2):
    """Covariance (mxx, myy, mxy) of a round Gaussian of width ``sigma``
    after GalSim's shear (g1, g2): sigma^2 S S^T with
    S = [[1 + g1, g2], [g2, 1 - g1]] / sqrt(1 - |g|^2)."""
    gg = g1 * g1 + g2 * g2
    k = sigma * sigma / (1.0 - gg)
    return k * (1.0 + gg + 2.0 * g1), k * (1.0 + gg - 2.0 * g1), k * 2.0 * g2


@pytest.mark.parametrize("force_center, offset", [(True, 0.0), (False, 0.5)])
def test_gaussian_covariance_recovery(force_center, offset):
    """The fitted intrinsic covariance (mxx, myy, mxy) of noise-free,
    PSF-convolved elliptical Gaussians -- the profile the model is -- equals
    the input covariance, together with the flux and the centre.  With a
    free centre the fit starts half a pixel from the truth in x and y.
    Measured agreement is 4e-5 sigma^2 or better."""
    fix = load("ngmix_gausscov")
    scale = 0.2
    fitter = anacal.ngmix.GaussFit(
        scale=scale, sigma_arcsec=0.4, stamp_size=48,
        force_center=force_center,
    )
    catalog = []
    for cx, cy in GAUSSCOV_CENTERS:
        src = anacal.table.galNumber()
        src.model.x1.v = (cx + offset) * scale
        src.model.x2.v = (cy - offset) * scale
        src.x1_det = src.model.x1.v
        src.x2_det = src.model.x2.v
        catalog.append(src)
    result = fitter.process_cell(
        catalog=catalog,
        img_array=fix["gal"],
        psf_array=fix["psf"],
        prior=anacal.ngmix.modelPrior(),
        num_epochs=10,
        variance=1.0,
    )
    assert len(result) == len(GAUSSCOV_CASES)
    for rr, (sigma, g1, g2, flux), (cx, cy) in zip(
        result, GAUSSCOV_CASES, GAUSSCOV_CENTERS
    ):
        m = rr.model
        np.testing.assert_allclose(
            [m.mxx.v, m.myy.v, m.mxy.v],
            _gaussian_covariance(sigma, g1, g2),
            rtol=0, atol=2e-4 * sigma**2,
        )
        np.testing.assert_allclose(m.F.v, flux, rtol=1e-4)
        np.testing.assert_allclose(
            [m.x1.v / scale, m.x2.v / scale], [cx, cy], rtol=0, atol=1e-4
        )
