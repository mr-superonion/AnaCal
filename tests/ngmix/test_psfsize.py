"""anacal.ngmix PSF size: metadetect's fitgauss reconvolution Gaussian,
sigma = sqrt(T d / 2), from a least-squares (= adaptive-moment) Gaussian
fit of the raw PSF image."""
import anacal
import numpy as np
import pytest

SCALE = 0.2


def _gauss_image(cxx, cyy, cxy, n=64, off=(0.3, -0.2)):
    y, x = np.mgrid[0:n, 0:n] * SCALE
    dx = x - (n // 2 + off[0]) * SCALE
    dy = y - (n // 2 + off[1]) * SCALE
    det = cxx * cyy - cxy**2
    return np.exp(
        -0.5 * (cyy * dx * dx - 2 * cxy * dx * dy + cxx * dy * dy) / det
    )


def _metadetect_dilation(e1, e2, T, dmax=1.1):
    # metadetect/lsst/metacal_exposures._get_ellip_dilation, with the
    # covariance built from (e1, e2, T) as ngmix.moments.e2mom does
    irr = 0.5 * T * (1 - e1)
    icc = 0.5 * T * (1 + e1)
    irc = 0.5 * T * e2
    eigs = np.linalg.eigvalsh(np.array([[irr, irc], [irc, icc]]))
    d = np.sqrt(eigs.max() / (T / 2.0))
    return min(1.0 + 2 * (d - 1.0), dmax)


def test_fit_recovers_elliptical_gaussian():
    cxx, cyy, cxy = 0.21, 0.15, 0.03
    r = anacal.ngmix.fit_psf_gauss(_gauss_image(cxx, cyy, cxy), SCALE)
    T = cxx + cyy
    assert r.converged
    np.testing.assert_allclose(r.T, T, rtol=1e-6)
    np.testing.assert_allclose(r.e1, (cxx - cyy) / T, atol=1e-7)
    np.testing.assert_allclose(r.e2, 2 * cxy / T, atol=1e-7)
    np.testing.assert_allclose(r.x1 / SCALE, 32.3, atol=1e-6)
    np.testing.assert_allclose(r.x2 / SCALE, 31.8, atol=1e-6)


@pytest.mark.parametrize(
    "e", [(0.0, 0.0), (0.03, -0.02), (0.1, 0.05), (0.2, 0.3)]
)
def test_dilation_matches_metadetect(e):
    for dmax in (1.1, 1.05):
        np.testing.assert_allclose(
            anacal.ngmix.ellip_dilation(e[0], e[1], dmax),
            _metadetect_dilation(e[0], e[1], 0.4, dmax),
            rtol=1e-12,
        )


def test_sigma_round_and_elliptical():
    # round PSF: d = 1 and sigma is the PSF's own Gaussian sigma
    s = anacal.ngmix.fitgauss_sigma_arcsec(_gauss_image(0.18, 0.18, 0.0), SCALE)
    np.testing.assert_allclose(s, np.sqrt(0.18), rtol=1e-6)
    # elliptical: sqrt(T d / 2) with metadetect's d
    cxx, cyy, cxy = 0.21, 0.15, 0.03
    T = cxx + cyy
    d = _metadetect_dilation((cxx - cyy) / T, 2 * cxy / T, T)
    s = anacal.ngmix.fitgauss_sigma_arcsec(_gauss_image(cxx, cyy, cxy), SCALE)
    np.testing.assert_allclose(s, np.sqrt(T * d / 2), rtol=1e-6)
    # the cap
    s_cap = anacal.ngmix.fitgauss_sigma_arcsec(
        _gauss_image(cxx, cyy, cxy), SCALE, dilation_max=1.0
    )
    np.testing.assert_allclose(s_cap, np.sqrt(T / 2), rtol=1e-6)


def test_non_gaussian_psf_is_adaptive_moments():
    # a Moffat-like PSF: the least-squares Gaussian is the adaptive-moment
    # fixed point, W = G(C) with <x x^T>_{I W} = C / 2
    n = 64
    y, x = np.mgrid[0:n, 0:n] * SCALE
    dx, dy = x - (n // 2) * SCALE, y - (n // 2) * SCALE
    q = (1.2 * dx * dx + 0.8 * dy * dy + 0.3 * dx * dy) / 0.35
    im = (1 + q) ** -2.5
    r = anacal.ngmix.fit_psf_gauss(im, SCALE)
    assert r.converged
    cxx = 0.5 * r.T * (1 + r.e1)
    cyy = 0.5 * r.T * (1 - r.e1)
    cxy = 0.5 * r.T * r.e2
    C = np.array([[cxx, cxy], [cxy, cyy]])
    Ci = np.linalg.inv(C)
    ex, ey = dx - (r.x1 - (n // 2) * SCALE), dy - (r.x2 - (n // 2) * SCALE)
    w = np.exp(
        -0.5 * (Ci[0, 0] * ex**2 + 2 * Ci[0, 1] * ex * ey + Ci[1, 1] * ey**2)
    )
    f = w * im
    mom = np.array([
        [np.sum(f * ex * ex), np.sum(f * ex * ey)],
        [np.sum(f * ex * ey), np.sum(f * ey * ey)],
    ]) / np.sum(f)
    np.testing.assert_allclose(mom, C / 2, atol=1e-5 * r.T)


def test_bad_input():
    with pytest.raises(ValueError):
        anacal.ngmix.fit_psf_gauss(np.zeros((3, 4, 4)), SCALE)
    with pytest.raises(ValueError):
        anacal.ngmix.ellip_dilation(0.1, 0.0, 0.9)
