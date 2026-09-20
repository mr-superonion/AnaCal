"""The single-Gaussian model in its covariance parameterisation.

The analytic derivatives with respect to the intrinsic covariance
(mxx, myy, mxy), the flux and the centre are pinned to values computed
with JAX (``jax.grad`` of the same model written in plain array
algebra; the script is at the bottom of this file), and also checked
against central finite differences of the model itself.  The test
point is the one the original (a1, a2, t) version used, a1 = 0.15,
a2 = 0.22, t = pi / 5, expressed in covariance components.
"""
import anacal
import numpy as np

SCALE = 1.0
SIGMA = 0.45

# the (a1, a2, t) = (0.15, 0.22, pi/5) source as intrinsic covariance
FLUX = 1.4
MXX = 0.03144822992284443
MYY = 0.03945177007715557
MXY = -0.01231618188602224
X1 = 0.15
X2 = -0.41


def _model(flux, mxx, myy, mxy, x1, x2):
    m = anacal.ngmix.NgmixGaussian()
    m.F = anacal.math.qnumber(flux)
    m.mxx = anacal.math.qnumber(mxx)
    m.myy = anacal.math.qnumber(myy)
    m.mxy = anacal.math.qnumber(mxy)
    m.x1 = anacal.math.qnumber(x1)
    m.x2 = anacal.math.qnumber(x2)
    return m


def test_ngmix_gaussian_against_jax():
    """r^2, the normalisation f and the model, with their gradients in
    (F, mxx, myy, mxy, x1, x2), against jax.grad (see the script
    below)."""
    gauss_model = _model(FLUX, MXX, MYY, MXY, X1, X2)
    kernel = gauss_model.prepare_modelD(scale=SCALE, sigma_arcsec=SIGMA)

    x = -0.43
    y = 0.21
    a = gauss_model.get_r2(x, y, kernel)
    res = np.array([
        a.v.v, a.v_F.v, a.v_mxx.v, a.v_myy.v, a.v_mxy.v, a.v_x1.v, a.v_x2.v
    ])
    res_target = np.array([
        2.8778969403, 0.0, -5.5252160456, -5.9674752241, 11.4841786576,
        4.7011556220, -4.8856832579,
    ])
    np.testing.assert_almost_equal(res, res_target)

    res = np.array(
        [kernel.f.v, kernel.f_mxx.v, kernel.f_myy.v, kernel.f_mxy.v]
    )
    res_target = np.array(
        [0.6698515119, -1.4354701651, -1.3879861434, -0.1461407919]
    )
    np.testing.assert_almost_equal(res, res_target)

    a = gauss_model.get_model(x, y, kernel)
    res = np.array([
        a.v.v, a.v_F.v, a.v_mxx.v, a.v_myy.v, a.v_mxy.v, a.v_x1.v, a.v_x2.v
    ])
    res_target = np.array([
        0.2224227455, 0.1588733897, 0.1378220981, 0.2027733151,
        -1.3256970074, -0.5228219703, 0.5433435420,
    ])
    np.testing.assert_almost_equal(res, res_target)
    return


# jax.grad of the model at the pixel (-0.43, 0.21), in the order
# (F, mxx, myy, mxy, x1, x2); same values as pinned above
_JAX_MODEL = 0.2224227455
_JAX_GRAD = np.array([
    0.1588733897, 0.1378220981, 0.2027733151, -1.3256970074,
    -0.5228219703, 0.5433435420,
])
_PAR_NAMES = ["F", "mxx", "myy", "mxy", "x1", "x2"]


def _loss_gradient(loss):
    return np.array([
        getattr(loss, "v_" + n).v for n in _PAR_NAMES
    ])


def _loss_curvature(loss):
    """The 6x6 Gauss-Newton matrix from the lossNumber fields, with
    every off-diagonal entry read from its own field."""
    pairs = {
        ("F", "F"): "v_FF", ("F", "mxx"): "v_Fmxx", ("F", "myy"): "v_Fmyy",
        ("F", "mxy"): "v_Fmxy", ("F", "x1"): "v_Fx1", ("F", "x2"): "v_Fx2",
        ("mxx", "mxx"): "v_mxxmxx", ("mxx", "myy"): "v_mxxmyy",
        ("mxx", "mxy"): "v_mxxmxy", ("mxx", "x1"): "v_mxxx1",
        ("mxx", "x2"): "v_mxxx2", ("myy", "myy"): "v_myymyy",
        ("myy", "mxy"): "v_myymxy", ("myy", "x1"): "v_myyx1",
        ("myy", "x2"): "v_myyx2", ("mxy", "mxy"): "v_mxymxy",
        ("mxy", "x1"): "v_mxyx1", ("mxy", "x2"): "v_mxyx2",
        ("x1", "x1"): "v_x1x1", ("x1", "x2"): "v_x1x2", ("x2", "x2"): "v_x2x2",
    }
    h = np.zeros((6, 6))
    for (a, b), field in pairs.items():
        i, j = _PAR_NAMES.index(a), _PAR_NAMES.index(b)
        h[i, j] = h[j, i] = getattr(loss, field).v
    return h


def test_loss_gradient_and_curvature_against_jax():
    """get_loss accumulates, per pixel, the chi2 gradient
    b_i = -2 (d - m) dm/dp_i / var and the Gauss-Newton curvature
    H_ij = 2 (dm/dp_i)(dm/dp_j) / var, every cross term included.  Both
    are checked against the JAX-pinned model gradient at the same
    pixel, for two pixel values (one on each side of the model)."""
    m = _model(FLUX, MXX, MYY, MXY, X1, X2)
    kernel = m.prepare_modelD(SCALE, SIGMA)
    x, y, var = -0.43, 0.21, 0.3
    r2 = m.get_r2(x, y, kernel)
    for d in (0.5, 0.1):
        loss = m.get_loss(anacal.math.qnumber(d), var, r2, kernel)
        resid = d - _JAX_MODEL
        np.testing.assert_allclose(loss.v.v, resid**2 / var, rtol=1e-8)
        np.testing.assert_allclose(loss.v_in.v, loss.v.v, rtol=0)
        np.testing.assert_allclose(
            _loss_gradient(loss), -2.0 * resid * _JAX_GRAD / var,
            rtol=1e-7, err_msg="chi2 gradient",
        )
        np.testing.assert_allclose(
            _loss_curvature(loss), 2.0 * np.outer(_JAX_GRAD, _JAX_GRAD) / var,
            rtol=1e-7, err_msg="Gauss-Newton curvature (all 21 entries)",
        )
        assert loss.n_pix == 1.0


def _stamp_pixels():
    xs = np.arange(-2.0, 2.01, 0.5)
    return [(x, y) for x in xs for y in xs]


def _chi2_gradient(pars, data, var):
    """Sum of the per-pixel get_loss gradients over a stamp."""
    m = _model(*pars)
    kernel = m.prepare_modelD(SCALE, SIGMA)
    g = np.zeros(6)
    for (x, y), d in zip(_stamp_pixels(), data):
        loss = m.get_loss(
            anacal.math.qnumber(d), var, m.get_r2(x, y, kernel), kernel
        )
        g += _loss_gradient(loss)
    return g


def test_loss_curvature_is_hessian_at_zero_residual():
    """Independent of JAX: on a stamp whose data equal the model, the
    exact Hessian of chi2 is the Gauss-Newton matrix (the residual term
    vanishes), so the summed curvature, cross terms included, must
    match central finite differences of the summed gradient; and the
    summed gradient must match finite differences of chi2 itself for
    data that do not equal the model."""
    pars = [FLUX, MXX, MYY, MXY, X1, X2]
    var = 0.3
    m = _model(*pars)
    kernel = m.prepare_modelD(SCALE, SIGMA)
    model_data = np.array([
        m.get_model(x, y, kernel).v.v for x, y in _stamp_pixels()
    ])
    h_an = np.zeros((6, 6))
    for (x, y), d in zip(_stamp_pixels(), model_data):
        loss = m.get_loss(
            anacal.math.qnumber(d), var, m.get_r2(x, y, kernel), kernel
        )
        h_an += _loss_curvature(loss)
    h = 1e-5
    h_fd = np.zeros((6, 6))
    for k in range(6):
        up, dn = list(pars), list(pars)
        up[k] += h
        dn[k] -= h
        h_fd[:, k] = (
            _chi2_gradient(up, model_data, var)
            - _chi2_gradient(dn, model_data, var)
        ) / (2.0 * h)
    np.testing.assert_allclose(h_fd, h_fd.T, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(
        h_an, h_fd, rtol=2e-5, atol=1e-8,
        err_msg="summed Gauss-Newton curvature vs FD Hessian",
    )

    # gradient against finite differences of chi2 on perturbed data
    data = model_data * 1.3 + 0.05

    def chi2(p):
        mm = _model(*p)
        kk = mm.prepare_modelD(SCALE, SIGMA)
        return sum(
            (d - mm.get_model(x, y, kk).v.v) ** 2 / var
            for (x, y), d in zip(_stamp_pixels(), data)
        )

    g_an = _chi2_gradient(pars, data, var)
    for k in range(6):
        up, dn = list(pars), list(pars)
        up[k] += h
        dn[k] -= h
        np.testing.assert_allclose(
            g_an[k], (chi2(up) - chi2(dn)) / (2.0 * h), rtol=1e-6,
            err_msg=f"d chi2 / d {_PAR_NAMES[k]}",
        )


def _value(pars, x, y):
    m = _model(*pars)
    return m.get_model(x, y, m.prepare_modelD(SCALE, SIGMA)).v.v


def _r2_value(pars, x, y):
    m = _model(*pars)
    return m.get_r2(x, y, m.prepare_modelD(SCALE, SIGMA)).v.v


def _fd(func, pars, k, x, y, h=1e-6):
    up = list(pars)
    dn = list(pars)
    up[k] += h
    dn[k] -= h
    return (func(up, x, y) - func(dn, x, y)) / (2.0 * h)


def test_model_derivatives_match_finite_differences():
    """The same derivatives at three more pixels, against central finite
    differences of the model itself."""
    pars = [FLUX, MXX, MYY, MXY, X1, X2]
    names = ["F", "mxx", "myy", "mxy", "x1", "x2"]
    for x, y in [(-0.43, 0.21), (0.9, -1.3), (0.15, -0.41)]:
        m = _model(*pars)
        kernel = m.prepare_modelD(SCALE, SIGMA)
        res = m.get_model(x, y, kernel)
        analytic = [
            res.v_F.v, res.v_mxx.v, res.v_myy.v, res.v_mxy.v,
            res.v_x1.v, res.v_x2.v,
        ]
        for k, (name, an) in enumerate(zip(names, analytic)):
            fd = _fd(_value, pars, k, x, y)
            np.testing.assert_allclose(
                an, fd, rtol=1e-6, atol=1e-9, err_msg=f"d model / d {name}"
            )
        r2 = m.get_r2(x, y, kernel)
        r2_an = [r2.v_mxx.v, r2.v_myy.v, r2.v_mxy.v, r2.v_x1.v, r2.v_x2.v]
        for k, (name, an) in enumerate(zip(names[1:], r2_an), start=1):
            fd = _fd(_r2_value, pars, k, x, y)
            np.testing.assert_allclose(
                an, fd, rtol=1e-6, atol=1e-9, err_msg=f"d r2 / d {name}"
            )


def test_model_value_and_normalisation():
    """m = F scale^2 / (2 pi sqrt(det C)) exp(-x^T C^-1 x / 2)."""
    pars = [FLUX, MXX, MYY, MXY, X1, X2]
    m = _model(*pars)
    kernel = m.prepare_modelD(SCALE, SIGMA)
    C = np.array([
        [pars[1] + SIGMA ** 2, pars[3]], [pars[3], pars[2] + SIGMA ** 2]
    ])
    Ci = np.linalg.inv(C)
    np.testing.assert_allclose(
        [kernel.ixx.v, kernel.ixy.v, kernel.iyy.v],
        [Ci[0, 0], Ci[0, 1], Ci[1, 1]],
    )
    np.testing.assert_allclose(
        kernel.f.v, SCALE ** 2 / (2 * np.pi * np.sqrt(np.linalg.det(C)))
    )
    x, y = -0.43, 0.21
    d = np.array([x - pars[4], y - pars[5]])
    expected = pars[0] * kernel.f.v * np.exp(-0.5 * d @ Ci @ d)
    np.testing.assert_allclose(_value(pars, x, y), expected, rtol=1e-12)


def test_axes_round_trip_and_shape():
    a1, a2, t = 0.15, 0.22, np.pi / 5.0
    m = anacal.ngmix.NgmixGaussian()
    m.set_axes(
        anacal.math.qnumber(a1),
        anacal.math.qnumber(a2),
        anacal.math.qnumber(t),
    )
    # this is the test point above
    np.testing.assert_allclose(
        [m.mxx.v, m.myy.v, m.mxy.v], [MXX, MYY, MXY], rtol=1e-12
    )
    # a1 is the semi-axis ALONG t; get_axes returns the major axis first
    ax = m.get_axes()
    np.testing.assert_allclose(
        [ax[0].v, ax[1].v, ax[2].v], [a2, a1, t - np.pi / 2.0], rtol=1e-8
    )
    e1, e2 = m.get_shape()
    e = (a1 ** 2 - a2 ** 2) / (a1 ** 2 + a2 ** 2)
    np.testing.assert_allclose(
        [e1.v, e2.v], [e * np.cos(2 * t), e * np.sin(2 * t)], rtol=1e-8
    )


def test_round_source_is_regular():
    """At exact roundness every derivative is finite and the shape is
    zero with a well-defined (zero) response -- the reason for fitting
    the covariance components rather than semi-axes and angle."""
    m = _model(1.0, 0.04, 0.04, 0.0, 0.0, 0.0)
    kernel = m.prepare_modelD(SCALE, SIGMA)
    res = m.get_model(0.3, -0.2, kernel)
    for q in (
        res.v, res.v_F, res.v_mxx, res.v_myy, res.v_mxy, res.v_x1, res.v_x2
    ):
        assert np.all(np.isfinite(q.to_array()))
    e1, e2 = m.get_shape()
    assert e1.v == 0.0 and e2.v == 0.0
    assert np.all(np.isfinite(e1.to_array()))
    assert np.all(np.isfinite(e2.to_array()))
    ax = m.get_axes()
    assert np.all(np.isfinite(ax[0].to_array()))


# The reference values in test_ngmix_gaussian_against_jax were produced
# by this script (jax 0.11, float64).  NOTE: jax segfaults inside the
# LSST "image" conda environment; run it in a plain venv with
# ``pip install "jax[cpu]"``.
#
# import jax, jax.numpy as jnp, numpy as np
# jax.config.update("jax_enable_x64", True)
# flux, t, a1, a2, x1, x2 = 1.4, np.pi / 5.0, 0.15, 0.22, 0.15, -0.41
# sigma, scale, x, y = 0.45, 1.0, -0.43, 0.21
# c, s = np.cos(t), np.sin(t)
# mxx = a1**2 * c**2 + a2**2 * s**2
# myy = a1**2 * s**2 + a2**2 * c**2
# mxy = (a1**2 - a2**2) * c * s
#
# def cov(d):
#     return jnp.array([[d[1] + sigma**2, d[3]], [d[3], d[2] + sigma**2]])
#
# def get_r2(d):
#     v = jnp.array([x - d[4], y - d[5]])
#     return v @ jnp.linalg.inv(cov(d)) @ v
#
# def get_f(d):
#     return scale**2 / jnp.sqrt(jnp.linalg.det(cov(d))) / (2.0 * jnp.pi)
#
# def get_model(d):
#     return d[0] * get_f(d) * jnp.exp(-0.5 * get_r2(d))
#
# data = jnp.array([flux, mxx, myy, mxy, x1, x2])
# print(np.array([get_r2(data)] + list(jax.grad(get_r2)(data))))
# print(np.array([get_f(data)] + list(jax.grad(get_f)(data)[1:4])))
# print(np.array([get_model(data)] + list(jax.grad(get_model)(data))))
