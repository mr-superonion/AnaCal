"""The smooth convergence gate, flux profiling and the apodised model.

The step of epoch k is scaled by a C1 smoothstep of the chi2 decrease
the step of epoch k-1 achieved, RELATIVE to the source's own chi2
scale F^2 H_FF / 2, from conv_tol (gate exactly 0: the source stops)
to conv_tol * gate_ratio (the plain step).  chi2 and the gate are
qnumbers, so the estimator is a smooth function of the pixels and the
propagated response is its exact derivative, whichever epoch a source
stops at.  The DEFAULT is conv_tol = 0, gate off, every source taking
exactly num_epochs epochs; the tests below switch the gate on
explicitly.
"""
import anacal
import numpy as np
import pytest

from ..fixtures import load

SCALE = 0.2
SIGMA = 0.4
FIX = load("ngmix_gaussfit")
# true centre of the fixture galaxies, in pixels from the stamp centre
# (the shift is applied after the shear, so it is the same for the
# whole shear pair)
DX1 = -0.2
DX2 = 0.11
E1_CONVERGED = 0.04418555  # e1 of gal_g1p_f150 after 40 ungated epochs


def _fit(img, num_epochs, variance=0.1, force_center=False, **kw):
    """One source on a fixture stamp.  With ``force_center`` the centre
    is held at the truth; otherwise it starts at the stamp centre and
    is fitted."""
    fitter = anacal.ngmix.GaussFit(
        scale=SCALE, sigma_arcsec=SIGMA, stamp_size=32,
        force_center=force_center, **kw
    )
    src = anacal.table.galNumber()
    x1 = (32 + DX1) * SCALE if force_center else 32 * SCALE
    x2 = (32 + DX2) * SCALE if force_center else 32 * SCALE
    src.model.x1.v = x1
    src.model.x2.v = x2
    src.x1_det = x1
    src.x2_det = x2
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
        q.to_array() for q in (m.F, m.mxx, m.myy, m.mxy, m.x1, m.x2)
    ])


def _pair(num_epochs, minus="gal_g1m_f300", var_minus=1.0, **kw):
    """The +/- 0.02 shear pair: finite difference of e1 across the pair
    against the mean propagated de1/dg1, and the multiplicative bias of
    the recovered shear."""
    c1 = _fit(FIX["gal_g1p_f150"], num_epochs, variance=0.1, **kw)
    c2 = _fit(FIX[minus], num_epochs, variance=var_minus, **kw)
    e1 = c1.model.get_shape()[0]
    e2 = c2.model.get_shape()[0]
    fd = (e1.v - e2.v) / 0.04
    an = 0.5 * (e1.g1 + e2.g1)
    return fd, an, fd / an - 1.0, c1, c2


@pytest.mark.parametrize(
    "conv_tol", [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
)
def test_gate_closes_at_every_tolerance(conv_tol):
    """The gate closes in finitely many epochs whatever the tolerance
    (a scaled step achieves a smaller decrease, which lowers the next
    gate), a tighter tolerance takes at least as many epochs, and the
    stopped e1 is within ~10 sqrt(tol) (relative) of the converged one:
    a relative chi2 change of tol is a relative parameter error of
    sqrt(tol)."""
    c = _fit(FIX["gal_g1p_f150"], 200, conv_tol=conv_tol)
    assert c.converged
    assert 1 < c.n_epochs < 12
    tighter = _fit(FIX["gal_g1p_f150"], 200, conv_tol=conv_tol * 1e-2)
    assert tighter.n_epochs >= c.n_epochs
    e1 = c.model.get_shape()[0].v
    assert abs(e1 - E1_CONVERGED) < 10.0 * np.sqrt(conv_tol) * E1_CONVERGED


def test_stop_is_a_stop():
    """Once the gate is exactly zero more epochs change nothing at all:
    the run with the large cap is bit-identical to a run capped at the
    epoch it stopped on, and one epoch short is a different fit."""
    long = _fit(FIX["gal_g1p_f150"], 200, conv_tol=1e-8)
    assert long.converged
    assert 0 < long.n_epochs < 200
    exact = _fit(FIX["gal_g1p_f150"], int(long.n_epochs), conv_tol=1e-8)
    assert exact.n_epochs == long.n_epochs
    np.testing.assert_array_equal(_params(long), _params(exact))
    short = _fit(FIX["gal_g1p_f150"], int(long.n_epochs) - 1, conv_tol=1e-8)
    assert not short.converged
    assert np.any(_params(short) != _params(long))


def test_gated_response_matches_finite_difference():
    """With the gate on at a relative tolerance of 1e-8 the pair stops
    through the gate (6 epochs) and the propagated response is the
    derivative of the gated estimator: the multiplicative bias of the
    recovered shear is below 1e-4 (measured 1e-5)."""
    fd, an, m, c1, c2 = _pair(30, conv_tol=1e-8)
    assert c1.converged and c2.converged
    assert abs(m) < 1.0e-4


@pytest.mark.parametrize("conv_tol", [1e-2, 1e-3, 1e-4])
def test_loose_tolerance_with_fixed_centre(conv_tol):
    """Loose relative tolerances (a relative chi2 change of 1e-2 is a
    relative parameter error of 0.1), on a same-flux shear pair with the
    centre held at the truth (a free, partially converged centre would
    add a centroid term that the pair does not cancel).  The gate closes
    after two epochs and the multiplicative bias of the recovered shear
    stays below 1e-4 (measured 2.4e-5)."""
    fd, an, m, c1, c2 = _pair(
        30, minus="gal_g1m_f150", var_minus=0.1,
        force_center=True, conv_tol=conv_tol,
    )
    assert c1.converged and c2.converged
    assert c1.n_epochs < 5 and c2.n_epochs < 5
    assert abs(m) < 1.0e-4


def test_default_is_fixed_epochs():
    """The default (conv_tol = 0) is the fixed-epoch mode: no source is
    flagged converged and every one takes exactly num_epochs epochs."""
    c = _fit(FIX["gal_g1p_f150"], 12)
    assert not c.converged and c.n_epochs == 12


@pytest.mark.parametrize("num_epochs", [3, 4, 30])
def test_response_is_exact_at_any_fixed_epoch_count(num_epochs):
    """Without the gate the propagated de1/dg1 is the derivative of the
    N-epoch estimator whether or not N epochs converge it.  The finite
    difference across the +/- 0.02 pair equals that derivative up to
    the O(g^2) curvature of the N-epoch map (3% at N = 1, 0.4% at
    N = 2, 2.5e-4 at N = 3, 1e-5 from N = 4 on)."""
    fd, an, m, _, _ = _pair(num_epochs)
    assert abs(m) < 1.0e-3


def test_flux_is_the_profiled_optimum():
    """After the fit the flux equals sum(d m~) / sum(m~^2) for the
    fitted shape, computed independently here on the deconvolved image."""
    cat = _fit(FIX["gal_g1p_f150"], 30)
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
    cat = _fit(FIX["gal_g1p_f150"], 30)
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


def test_ellipticity_prior_response_is_exact():
    """A Gaussian prior on (e1, e2) at the re-smoothing scale shrinks e
    and its response together, and the propagated response stays the
    exact derivative: the multiplicative bias of the same-flux pair is
    unchanged at 1e-6 with the prior on, while e1 and R both drop."""
    def pair(sigma_e):
        prior = anacal.ngmix.modelPrior()
        if sigma_e > 0:
            prior.set_sigma_e(anacal.math.qnumber(sigma_e))
        out = []
        for key in ("gal_g1p_f150", "gal_g1m_f150"):
            fitter = anacal.ngmix.GaussFit(
                scale=SCALE, sigma_arcsec=SIGMA, stamp_size=32
            )
            src = anacal.table.galNumber()
            src.model.x1.v = src.model.x2.v = 32 * SCALE
            src.x1_det = src.x2_det = 32 * SCALE
            out.append(fitter.process_cell(
                catalog=[src], img_array=FIX[key], psf_array=FIX["psf"],
                prior=prior, num_epochs=30, variance=0.1,
            )[0].model.get_shape()[0])
        e_p, e_m = out
        fd = (e_p.v - e_m.v) / 0.04
        an = 0.5 * (e_p.g1 + e_m.g1)
        return e_p.v, an, fd / an - 1.0

    e0, r0, m0 = pair(0.0)
    e1, r1, m1 = pair(0.1)
    assert abs(m0) < 1.0e-4 and abs(m1) < 1.0e-4
    assert 0.0 < e1 < e0 and 0.0 < r1 < r0
    # shrinkage of e and of R go together, so e / R is what it was
    np.testing.assert_allclose(e1 / r1, e0 / r0, rtol=2.0e-3)
