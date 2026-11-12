"""The adaptive kernel (afpfs1): every source measured with its own
kernel size from the detector's Gaussian fit, interpolated on a
tabulated basis, with the size's shear response folded into every
response.

Checks: a source whose size sits exactly on a grid node reproduces the
fixed kernel at that size; the size derivative that enters the
responses is the slope of the interpolant actually evaluated (finite
difference in sigma); the size map's own derivative; the error paths.
The shear accuracy of the estimator is tested with the ring test of
test_fpfs_shearest.py.
"""

import anacal
import numpy as np
import pytest

from ..fixtures import cosmos_centres, cosmos_scene

SCALE = 0.2
NOISE_STD = 0.37
SIGMA = 0.52
FIT = anacal.fpfs.ADAPTIVE_FIT_COLUMNS
# the mode-noise family is summed in a different order in the table
# (m00 alone) than in the fixed kernel's full covariance
NOISE_FAMILY = ("m00_err", "flux_err", "s2n", "ds2n_dg1", "ds2n_dg2")


def _scene(seed=1):
    gal, psf = cosmos_scene()
    rng = np.random.RandomState(seed)
    img = gal + rng.normal(scale=NOISE_STD, size=gal.shape)
    noise = rng.normal(scale=NOISE_STD, size=gal.shape)
    det0 = cosmos_centres()
    det = np.zeros(
        len(det0),
        dtype=[("y", "f8"), ("x", "f8")] + [(c, "f8") for c in FIT],
    )
    det["y"], det["x"] = det0["y"], det0["x"]
    # synthetic fit covariances with a response
    det["mxx"] = 0.15 + 0.2 * rng.rand(len(det))
    det["myy"] = 0.15 + 0.2 * rng.rand(len(det))
    det["dmxx_dg1"] = rng.normal(0.0, 0.3, len(det))
    det["dmyy_dg1"] = rng.normal(0.0, 0.3, len(det))
    det["dmxx_dg2"] = rng.normal(0.0, 0.3, len(det))
    det["dmyy_dg2"] = rng.normal(0.0, 0.3, len(det))
    return img, psf, noise, det


def _run(cfg, img, psf, noise, det):
    return anacal.fpfs.process_image(
        fpfs_config=cfg,
        mag_zero=30.0,
        gal_array=img,
        psf_array=psf,
        pixel_scale=SCALE,
        noise_variance=NOISE_STD**2,
        noise_array=noise,
        detection=det,
    )


def _col(kernel, name):
    return anacal.fpfs._kernel_rename_map([name], kernel)[name]


def test_node_matches_fixed_kernel(monkeypatch):
    """A source exactly on a grid node with no size response is the
    fixed kernel at that size: the interpolation weights are (0, 1, 0,
    0), so the modes and every shape are bit-identical."""
    img, psf, noise, det = _scene()
    grid = anacal.fpfs.adaptive_sigma_grid(SIGMA, 3 * SIGMA, 0.02)
    node = float(grid[6])

    def on_node(*args, **kwargs):
        n = len(det)
        return np.full(n, node), np.zeros(n), np.zeros(n)

    monkeypatch.setattr(anacal.fpfs, "adaptive_sigma", on_node)
    cfg = anacal.fpfs.FpfsConfig(sigma_shapelets1=SIGMA, adaptive_sigma=True)
    a = _run(cfg, img, psf, noise, det)
    assert np.all(a["afpfs1_sigma"] == node)
    assert np.all(a["afpfs1_dsigma_dg1"] == 0.0)
    fixed = anacal.fpfs.FpfsConfig(sigma_shapelets1=node)
    b = _run(fixed, img, psf, noise, det)
    for name in anacal.fpfs._catalog_columns:
        x, y = a[_col("afpfs1", name)], b[_col("fpfs1", name)]
        if name in NOISE_FAMILY:
            np.testing.assert_allclose(x, y, rtol=1e-12, err_msg=name)
        else:
            np.testing.assert_array_equal(x, y, err_msg=name)


def test_size_derivative_is_the_interpolant_slope(monkeypatch):
    """The dsigma/dg term of every response is the derivative of the
    evaluated (interpolated) quantity: unit size response minus none
    equals the finite difference of the value in sigma."""
    img, psf, noise, det = _scene()
    n = len(det)
    # sizes spread over the inner 60% of several grid intervals: the
    # cubic is C1 only, so a central difference across a node would
    # carry the jump of the second derivative
    rng = np.random.RandomState(3)
    grid = anacal.fpfs.adaptive_sigma_grid(SIGMA, 3 * SIGMA, 0.02)
    k = rng.randint(5, 15, n)
    sig0 = grid[k] * (grid[k + 1] / grid[k]) ** (0.2 + 0.6 * rng.rand(n))
    delta = 1e-4 * sig0
    cfg = anacal.fpfs.FpfsConfig(sigma_shapelets1=SIGMA, adaptive_sigma=True)

    def run_with(sig, ds1):
        monkeypatch.setattr(
            anacal.fpfs, "adaptive_sigma",
            lambda *a, **k: (sig, np.full(n, ds1), np.zeros(n)),
        )
        return _run(cfg, img, psf, noise, det)

    base = run_with(sig0, 0.0)
    unit = run_with(sig0, 1.0)
    plus = run_with(sig0 + delta, 0.0)
    minus = run_with(sig0 - delta, 0.0)
    assert np.all(unit["afpfs1_dsigma_dg1"] == 1.0)
    for value, resp in (
        ("m00", "dm00_dg1"), ("m20", "dm20_dg1"),
        ("m22c", "dm22c_dg1"), ("m22s", "dm22s_dg1"),
        ("e1", "de1_dg1"), ("e2", "de2_dg1"),
        ("flux", "dflux_dg1"), ("s2n", "ds2n_dg1"),
    ):
        v, r = _col("afpfs1", value), _col("afpfs1", resp)
        got = unit[r] - base[r]
        fd = (plus[v] - minus[v]) / (2.0 * delta)
        # the central difference's own truncation error, (delta / h)^2
        # of the variation over an interval, relative to the column
        np.testing.assert_allclose(
            got, fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max(), err_msg=value
        )
    # the fixed kernels are untouched by the size response
    for name in ("e1", "de1_dg1", "s2n"):
        np.testing.assert_array_equal(
            unit[_col("fpfs1", name)], base[_col("fpfs1", name)]
        )


def test_size_map_derivative():
    """adaptive_sigma's dsigma/dg is the chain rule through the clamps:
    unit dT/dg against the finite difference in T."""
    floor, cap = 0.5, 1.5
    tt = np.array([-0.3, 0.0, 1e-3, 0.1, 0.5, 2.0, 3.9, 4.5, 20.0])
    d = 1e-6
    sig, ds1, ds2 = anacal.fpfs.adaptive_sigma(
        tt / 2, tt / 2, 1.0, 0.0, 0.0, 0.0, floor, cap
    )
    sp = anacal.fpfs.adaptive_sigma(
        tt / 2 + d, tt / 2, 0, 0, 0, 0, floor, cap
    )[0]
    sm = anacal.fpfs.adaptive_sigma(
        tt / 2 - d, tt / 2, 0, 0, 0, 0, floor, cap
    )[0]
    np.testing.assert_allclose(ds1, (sp - sm) / (2 * d), rtol=1e-6, atol=1e-9)
    assert np.all(ds2 == 0.0)
    # the floor is strict (the basis table starts there); the cap is
    # smooth and exceeded only by the floor clamp's eps^2 / (4 |y - cap|)
    assert np.all(sig > floor) and np.all(sig < cap * (1.0 + 1e-6))
    # alpha = 1: sigma^2 = floor^2 + T / 2 away from the clamps
    mid = (tt > 0.3) & (tt < 2.0)
    np.testing.assert_allclose(
        sig[mid] ** 2, floor**2 + tt[mid] / 2, rtol=1e-4
    )


def test_second_adaptive_kernel():
    """afpfs2 runs only for adaptive_fpfs_kernel_ratio >= 1.1, with
    sigma^2 = floor^2 + ratio^2 T / 2 (same floor and cap)."""
    img, psf, noise, det = _scene()
    one = _run(
        anacal.fpfs.FpfsConfig(sigma_shapelets1=SIGMA, adaptive_sigma=True),
        img, psf, noise, det,
    )
    assert not any("afpfs2" in n for n in one.dtype.names)
    low = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=SIGMA, adaptive_sigma=True,
        adaptive_fpfs_kernel_ratio=1.05,
    )
    assert not any(
        "afpfs2" in n for n in _run(low, img, psf, noise, det).dtype.names
    )
    ratio = 1.5
    cfg = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=SIGMA, adaptive_sigma=True,
        adaptive_fpfs_kernel_ratio=ratio,
    )
    two = _run(cfg, img, psf, noise, det)
    assert len(two.dtype.names) == len(one.dtype.names) + len(
        anacal.fpfs._adaptive_columns
    )
    # afpfs1 is the same measurement with or without afpfs2
    for name in one.dtype.names:
        np.testing.assert_array_equal(two[name], one[name], err_msg=name)
    s1, s2 = two["afpfs1_sigma"], two["afpfs2_sigma"]
    assert np.all(s2 > s1)
    # away from the clamps the excess over the floor scales with ratio^2
    mid = (s1 > 1.05 * SIGMA) & (s2 < 2.5 * SIGMA)
    assert mid.sum() > 10
    np.testing.assert_allclose(
        (s2[mid] ** 2 - SIGMA**2) / (s1[mid] ** 2 - SIGMA**2),
        ratio**2, rtol=1e-3,
    )
    # and so does sigma dsigma/dg = d(sigma^2)/dg / 2
    np.testing.assert_allclose(
        (s2 * two["afpfs2_dsigma_dg1"])[mid],
        ratio**2 * (s1 * two["afpfs1_dsigma_dg1"])[mid], rtol=1e-3,
    )
    assert np.all(np.isfinite(two["afpfs2_e1"]))
    assert np.all(np.isfinite(two["afpfs2_de1_dg1"]))
    # afpfs2 needs the adaptive kernel on
    bad = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=SIGMA, adaptive_fpfs_kernel_ratio=ratio
    )
    with pytest.raises(ValueError, match="adaptive_sigma"):
        _run(bad, img, psf, noise, det)


def test_adaptive_errors():
    img, psf, noise, det = _scene()
    cfg = anacal.fpfs.FpfsConfig(sigma_shapelets1=SIGMA, adaptive_sigma=True)
    # the fit columns are required
    with pytest.raises(ValueError, match="fit"):
        _run(cfg, img, psf, noise, det[["y", "x"]].copy())
    # cap below the floor
    bad = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=SIGMA, adaptive_sigma=True, adaptive_sigma_max=0.3
    )
    with pytest.raises(ValueError, match="sigma_floor < sigma_max"):
        _run(bad, img, psf, noise, det)
    # with the option off nothing adaptive is measured
    plain = _run(
        anacal.fpfs.FpfsConfig(sigma_shapelets1=SIGMA), img, psf, noise, det
    )
    assert not any("afpfs" in n for n in plain.dtype.names)
    grid = anacal.fpfs.adaptive_sigma_grid(SIGMA, 3 * SIGMA, 0.02)
    assert grid[1] == SIGMA and grid[-2] >= 3 * SIGMA and len(grid) >= 4
