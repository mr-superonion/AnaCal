"""The fit's convergence stop, flux profiling and apodised model.

The fitter stops a source once its step -- value and shear-response
slots together -- falls below ``conv_tol`` (rmodel.h,
``update_model_params``); ``n_epochs`` records how far it went.
"""
import anacal
import numpy as np

from ..fixtures import load

SCALE = 0.2
SIGMA = 0.4
FIX = load("ngmix_gaussfit")


def _fit(img, num_epochs, variance=0.1, **kw):
    fitter = anacal.ngmix.GaussFit(
        scale=SCALE, sigma_arcsec=SIGMA, stamp_size=32, **kw
    )
    src = anacal.table.galNumber()
    src.model.x1.v = 32 * SCALE
    src.model.x2.v = 32 * SCALE
    src.x1_det = 32 * SCALE
    src.x2_det = 32 * SCALE
    return fitter.process_cell(
        catalog=[src],
        img_array=img,
        psf_array=FIX["psf"],
        prior=anacal.ngmix.modelPrior(),
        num_epochs=num_epochs,
        variance=variance,
    )[0]


def _params(cat):
    m = cat.model
    return np.array([
        q.to_array() for q in (m.F, m.t, m.a1, m.a2, m.x1, m.x2)
    ])


def test_stop_is_a_stop():
    """Once a source has stopped, more epochs change nothing at all.

    The run with the large cap must have stopped early, and its output
    must be bit-identical to a run capped exactly at the epoch it
    stopped on: the remaining epochs are skipped, not re-run.
    """
    long = _fit(FIX["gal_g1p_f150"], 200)
    assert long.converged
    assert 0 < long.n_epochs < 200
    exact = _fit(FIX["gal_g1p_f150"], int(long.n_epochs))
    assert exact.n_epochs == long.n_epochs
    np.testing.assert_array_equal(_params(long), _params(exact))
    # one epoch short is NOT the same fit: the last step was real
    short = _fit(FIX["gal_g1p_f150"], int(long.n_epochs) - 1)
    assert not short.converged
    assert np.any(_params(short) != _params(long))


def test_last_step_is_below_tolerance():
    """The jump a stop can introduce is bounded by the tolerance.

    The estimator is 'n epochs' with n data-dependent, so it can jump by
    the size of the final step where n changes; that step, in every
    qnumber slot, must be below sqrt(2 tol / curv) for the stiffest
    parameter -- here the centroid.
    """
    tol = 1.0e-10
    long = _fit(FIX["gal_g1p_f150"], 200, conv_tol=tol)
    short = _fit(FIX["gal_g1p_f150"], int(long.n_epochs) - 1, conv_tol=tol)
    jump = np.abs(_params(long) - _params(short))
    # curvatures are chi2 per arcsec^2 / rad^2; the flux is profiled
    # exactly so its jump follows the shape's
    curv = min(
        long.loss.v_mxxmxx.v, long.loss.v_myymyy.v, long.loss.v_mxymxy.v,
        long.loss.v_x1x1.v, long.loss.v_x2x2.v,
    )
    assert curv > 0
    bound = np.sqrt(2.0 * tol / curv)
    assert np.all(jump[1:] <= 10.0 * bound), (jump, bound)


def test_tighter_tolerance_takes_more_epochs():
    n = [
        _fit(FIX["gal_g1p_f150"], 200, conv_tol=t).n_epochs
        for t in (1.0e-4, 1.0e-6, 1.0e-8)
    ]
    assert n[0] <= n[1] <= n[2]
    assert n[0] < n[2]


def test_converged_fit_matches_finite_difference_response():
    """The propagated de1/dg1 against the estimator's own response to
    the +/- 0.02 shear pair, at the tolerance-defined stop (not at a
    fixed epoch count)."""
    c1 = _fit(FIX["gal_g1p_f150"], 200, variance=0.1)
    c2 = _fit(FIX["gal_g1m_f300"], 200, variance=1.0)
    assert c1.converged and c2.converged
    e1 = c1.model.get_shape()[0]
    e2 = c2.model.get_shape()[0]
    fd = (e1.v - e2.v) / 0.04
    an = 0.5 * (e1.g1 + e2.g1)
    assert abs(fd / an - 1.0) < 2.0e-3


def test_flux_is_the_profiled_optimum():
    """After the fit the flux equals sum(d m~) / sum(m~^2) for the
    fitted shape, computed independently here on the deconvolved image."""
    cat = _fit(FIX["gal_g1p_f150"], 200)
    img = FIX["gal_g1p_f150"]
    ny, nx = img.shape
    q = anacal.image.ImageQ(
        nx=nx, ny=ny, scale=SCALE, sigma_arcsec=SIGMA, klim=100.0
    )
    data = q.prepare_qnumber_image(
        img, FIX["psf"], xcen=nx // 2, ycen=ny // 2
    )[0]
    stamp = cat.model.get_image_stamp(nx, ny, SCALE, SIGMA)[0]
    # get_image_stamp is centred on the model's rounded pixel; the
    # fitted centre here is within a pixel of the stamp centre
    unit = stamp / cat.model.F.v
    yy, xx = np.mgrid[0:ny, 0:nx]
    win = (
        (xx * SCALE - cat.model.x1.v) ** 2 + (yy * SCALE - cat.model.x2.v) ** 2
    ) < 3.5 ** 2
    f_star = np.sum(data[win] * unit[win]) / np.sum(unit[win] ** 2)
    np.testing.assert_allclose(cat.model.F.v, f_star, rtol=1.0e-6, atol=0)


def test_model_is_apodised_to_exactly_zero():
    """Beyond r^2 = 25 the model is exactly zero, inside r^2 = 20 it is
    the plain Gaussian; in between it is monotonic."""
    cat = _fit(FIX["gal_g1p_f150"], 200)
    m = cat.model
    nx = ny = 96
    stamp = m.get_image_stamp(nx, ny, SCALE, SIGMA)[0]
    kb = m.prepare_modelD(SCALE, SIGMA)
    x0 = round(m.x1.v / SCALE) * SCALE
    y0 = round(m.x2.v / SCALE) * SCALE
    yy, xx = np.mgrid[0:ny, 0:nx]
    xs = (xx - nx // 2) * SCALE + x0
    ys = (yy - ny // 2) * SCALE + y0
    r2 = np.array([
        m.get_r2(float(x), float(y), kb).v.v
        for x, y in zip(xs.ravel(), ys.ravel())
    ]).reshape(ny, nx)
    assert np.all(stamp[r2 >= 25.0] == 0.0)
    inner = r2 < 20.0
    gauss = m.F.v * kb.f.v * np.exp(-0.5 * r2)
    np.testing.assert_allclose(stamp[inner], gauss[inner], rtol=1e-12)
    band = (r2 > 20.0) & (r2 < 25.0)
    assert np.all(stamp[band] > 0.0)
    assert np.all(stamp[band] < gauss[band])
