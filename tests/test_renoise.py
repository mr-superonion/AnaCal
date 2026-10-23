"""anacal.renoise: the spectral layer of the minimal-variance renoising.

Checks, against closed forms and against the pipeline's own pixel shear
response (ImageQ), that

* the 90-degree rotation about k = 0 and the k-derivatives of the noise
  spectrum and of the PSF power are exact on the grid;
* the filler removes the spin-2 part of the reconvolved noise spectrum;
* with the assigned response the spectrum of the total noise is
  independent of the shear to first order, i.e. the shear dependence of
  epsilon + gamma D epsilon is cancelled.
"""

import anacal
import numpy as np
import pytest

from .fixtures import symmetrized_spectrum

SCALE = 0.2
SIGMA_H = 0.52
KLIM = anacal.task.KLIM_PIX / SCALE


def kgrid(n):
    """(k1, k2) on the rfft half-plane, shape (n, n // 2 + 1)."""
    k1 = 2.0 * np.pi * np.fft.rfftfreq(n, d=SCALE)
    k2 = 2.0 * np.pi * np.fft.fftfreq(n, d=SCALE)
    return np.meshgrid(k1, k2)


def inband(n, margin=0):
    k1, k2 = kgrid(n)
    ok = k1**2 + k2**2 < (KLIM - margin) ** 2
    ok[n // 2, :] = False
    ok[:, n // 2] = False
    return ok


def gauss_psf(n, sxx, syy, sxy=0.0):
    """Elliptical Gaussian PSF, centred on pixel (n // 2, n // 2)."""
    y, x = (np.mgrid[0:n, 0:n] - n // 2) * SCALE
    det = sxx * syy - sxy**2
    q = (syy * x * x - 2.0 * sxy * x * y + sxx * y * y) / det
    psf = np.exp(-0.5 * q)
    return psf / psf.sum()


def gauss_psf_power(n, sxx, syy, sxy=0.0):
    """|p(k)|^2 of gauss_psf: exp(-k^T Sigma k)."""
    k1, k2 = kgrid(n)
    return np.exp(-(sxx * k1**2 + 2.0 * sxy * k1 * k2 + syy * k2**2))


def apply(filt, img):
    """Real-space image filtered by a half-plane kernel."""
    n = img.shape[0]
    return np.fft.irfft2(filt * np.fft.rfft2(img), s=(n, n))


def test_rotate90_is_exact_permutation():
    n = 32
    k1, k2 = kgrid(n)
    f = (
        lambda a, b: np.exp(-(0.05 * a * a + 0.02 * b * b)) + 0.01 * a * a
    )  # noqa: E731
    arr = f(k1, k2)
    rot = anacal.renoise.rotate90(arr)
    # (R a)(k1, k2) = a(-k2, k1); a is even, so the half plane suffices
    ok = inband(n, margin=-100)
    np.testing.assert_allclose(rot[ok], f(-k2, k1)[ok], rtol=0, atol=1e-14)
    # R^4 = 1
    r4 = arr
    for _ in range(4):
        r4 = anacal.renoise.rotate90(r4)
    np.testing.assert_allclose(r4[ok], arr[ok], atol=1e-14)


def test_noise_spectrum_white():
    ns = anacal.renoise.make_noise_spectrum(64, SCALE, 0.3)
    assert ns.white and ns.S.shape == (64, 33)
    assert np.all(ns.S == 0.3) and np.all(ns.S_R == 0.3)
    assert not ns.dS1.any() and not ns.dS2.any()


def test_noise_spectrum_gaussian_correlation():
    """C(x) = var exp(-x^T Sigma^-1 x / 2) has S = var 2 pi sqrt(det Sigma)
    / scale^2 exp(-k^T Sigma k / 2) and known log-derivatives."""
    n, var = 64, 0.7
    sxx, syy, sxy = 0.09, 0.05, 0.02
    m = 25
    y, x = (np.mgrid[0:m, 0:m] - m // 2) * SCALE
    det = sxx * syy - sxy**2
    corr = var * np.exp(
        -0.5 * (syy * x * x - 2 * sxy * x * y + sxx * y * y) / det
    )
    ns = anacal.renoise.make_noise_spectrum(n, SCALE, var, corr)
    k1, k2 = kgrid(n)
    S_true = (
        var
        * 2
        * np.pi
        * np.sqrt(det)
        / SCALE**2
        * np.exp(-0.5 * (sxx * k1**2 + 2 * sxy * k1 * k2 + syy * k2**2))
    )
    # exact against numpy's discrete transforms of the same sampled C
    big = np.zeros((n, n))
    big[
        n // 2 - m // 2 : n // 2 - m // 2 + m,
        n // 2 - m // 2 : n // 2 - m // 2 + m,
    ] = corr
    yy, xx = (np.mgrid[0:n, 0:n] - n // 2) * SCALE
    F = lambda arr: np.fft.rfft2(np.fft.ifftshift(arr))  # noqa: E731
    S_np = F(big).real
    d1, d2 = F(big * xx).imag, F(big * yy).imag
    np.testing.assert_allclose(ns.S, S_np, atol=1e-12 * S_np.max())
    np.testing.assert_allclose(
        ns.dS1, k1 * d1 - k2 * d2, atol=1e-12 * S_np.max()
    )
    np.testing.assert_allclose(
        ns.dS2, k1 * d2 + k2 * d1, atol=1e-12 * S_np.max()
    )
    # the continuous closed form holds where the sampled Gaussian is not
    # aliased, |k| well below the Nyquist frequency
    sel = (np.hypot(k1, k2) < 6.0) & (S_true > 1e-4 * S_true.max())
    np.testing.assert_allclose(ns.S[sel], S_true[sel], rtol=1e-6)
    # (k1 d1 - k2 d2) ln S and (k1 d2 + k2 d1) ln S
    a1 = -(sxx * k1**2 - syy * k2**2)
    a2 = -(sxy * (k1**2 + k2**2) + (sxx + syy) * k1 * k2)
    np.testing.assert_allclose((ns.dS1 / ns.S)[sel], a1[sel], atol=1e-5)
    np.testing.assert_allclose((ns.dS2 / ns.S)[sel], a2[sel], atol=1e-5)
    np.testing.assert_allclose(ns.S_R, anacal.renoise.rotate90(ns.S))


def sym_reference(sd, sd_rot, a, a_rot, delta):
    """numpy transcription of equations (fill) and (asymchain)."""
    d = sd - sd_rot
    sig = sd + sd_rot
    D = np.sqrt(d * d + delta**2 * sig * sig)
    ssym = 0.5 * (sig + D)
    dS, dSR = a * sd, -a_rot * sd_rot
    inv = np.where(D > 0, 1.0 / np.where(D > 0, D, 1.0), 0.0)
    asym = (
        0.5
        / ssym
        * (dS + dSR + (d * (dS - dSR) + delta**2 * sig * (dS + dSR)) * inv)
    )
    return ssym, asym


@pytest.mark.parametrize("delta", [0.0, 0.05])
def test_kernels_elliptical_gaussian_psf(delta):
    n, var, c = 64, 0.5, 0.1
    sxx, syy, sxy = 0.14, 0.10, 0.02
    ns = anacal.renoise.make_noise_spectrum(n, SCALE, var)
    rk = anacal.renoise.make_renoise_kernels(
        ns, gauss_psf(n, sxx, syy, sxy), KLIM, c, delta
    )
    k1, k2 = kgrid(n)
    ok = inband(n)
    psf = gauss_psf(n, sxx, syy, sxy)
    # discrete reference, from numpy transforms of the same sampled PSF
    yy, xx = (np.mgrid[0:n, 0:n] - n // 2) * SCALE
    F = lambda arr: np.fft.rfft2(np.fft.ifftshift(arr))  # noqa: E731
    pk, p1, p2 = F(psf), F(psf * xx), F(psf * yy)
    # the deconvolution floor of Image::deconvolve: |p| below 1e-5 p(0)
    # is replaced by the floor, where ln |p|^2 is then constant
    floor = 1e-5 * pk[0, 0].real
    low = np.abs(pk) < floor
    sd = var / np.where(low, floor, np.abs(pk)) ** 2
    a = np.where(low, 0.0, -2.0 * (k1 * (p1 / pk).imag - k2 * (p2 / pk).imag))
    a2 = np.where(low, 0.0, -2.0 * (k1 * (p2 / pk).imag + k2 * (p1 / pk).imag))
    np.testing.assert_allclose(rk.Sd[ok], sd[ok], rtol=1e-10)
    # symmetrized spectrum and a_sym by the chain rule, in numpy
    rot = anacal.renoise.rotate90
    ssym, asym1 = sym_reference(sd, rot(sd), a, rot(a), delta)
    _, asym2 = sym_reference(sd, rot(sd), a2, rot(a2), delta)
    np.testing.assert_allclose(rk.Sd_sym[ok], ssym[ok], rtol=1e-10)
    np.testing.assert_allclose(rk.a_sym1[ok], asym1[ok], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(rk.a_sym2[ok], asym2[ok], rtol=1e-8, atol=1e-10)
    # continuous closed form, where the sampled Gaussian is not aliased:
    # |p|^2 = exp(-k^T Sigma k), a = 2 (Sxx k1^2 - Syy k2^2), ...
    band = ok & (np.hypot(k1, k2) < 8.0)
    sd_c = var / gauss_psf_power(n, sxx, syy, sxy)
    np.testing.assert_allclose(rk.Sd[band], sd_c[band], rtol=1e-6)
    np.testing.assert_allclose(
        a[band], 2.0 * (sxx * k1**2 - syy * k2**2)[band], atol=1e-6
    )
    np.testing.assert_allclose(
        a2[band],
        2.0 * (sxy * (k1**2 + k2**2) + (sxx + syy) * k1 * k2)[band],
        atol=1e-6,
    )
    # exact symmetry and spin-2 oddness
    np.testing.assert_allclose(rot(rk.Sd_sym)[ok], rk.Sd_sym[ok], rtol=1e-12)
    np.testing.assert_allclose(rot(rk.a_sym1)[ok], -rk.a_sym1[ok], atol=1e-9)
    np.testing.assert_allclose(rot(rk.a_sym2)[ok], -rk.a_sym2[ok], atol=1e-9)
    # the filler raises S^d to S^d_sym, and the kernels are consistent
    assert np.all(rk.g >= 0)
    np.testing.assert_allclose(
        (rk.Sd * (1 + rk.g**2))[ok], rk.Sd_sym[ok], rtol=1e-10
    )
    ratio = rk.Sd_sym[ok] / rk.Sd[ok]  # S^d > 0 inside the band
    np.testing.assert_allclose((rk.w**2)[ok], c * ratio)
    np.testing.assert_allclose(
        rk.kappa1[ok],
        -np.sqrt(ratio) * rk.a_sym1[ok] / (2 * np.sqrt(c)),
    )
    if delta == 0.0:
        # finite differences of ln S^d_sym, away from the max crossing
        ssym_f = lambda q1, q2: 0.5 * (  # noqa: E731
            var / np.exp(-(sxx * q1**2 + 2 * sxy * q1 * q2 + syy * q2**2))
            + var / np.exp(-(sxx * q2**2 - 2 * sxy * q1 * q2 + syy * q1**2))
            + np.abs(
                var / np.exp(-(sxx * q1**2 + 2 * sxy * q1 * q2 + syy * q2**2))
                - var / np.exp(-(sxx * q2**2 - 2 * sxy * q1 * q2 + syy * q1**2))
            )
        )
        h = 1e-5
        fd = (
            k1 * (np.log(ssym_f(k1 + h, k2)) - np.log(ssym_f(k1 - h, k2)))
            - k2 * (np.log(ssym_f(k1, k2 + h)) - np.log(ssym_f(k1, k2 - h)))
        ) / (2 * h)
        far = band & (np.abs(sd - rot(sd)) > 0.05 * (sd + rot(sd)))
        np.testing.assert_allclose(
            rk.a_sym1[far], fd[far], rtol=1e-5, atol=1e-6
        )


def test_round_psf_needs_no_filler():
    n = 64
    ns = anacal.renoise.make_noise_spectrum(n, SCALE, 1.0)
    rk = anacal.renoise.make_renoise_kernels(
        ns, gauss_psf(n, 0.12, 0.12), KLIM, 0.1, 0.0
    )
    ok = inband(n)
    # |p(k)|^2 and |p(R k)|^2 of a sampled round Gaussian agree to FFT
    # roundoff, ~1e-11 relative just above the deconvolution floor; g is
    # the square root of that difference
    assert rk.g[ok].max() < 1e-4
    np.testing.assert_allclose(rk.w[ok], np.sqrt(0.1), rtol=1e-9)
    np.testing.assert_allclose(rk.Sd_sym[ok], rk.Sd[ok], rtol=1e-9)


def sector_power(P, n, kmin, kmax, half_angle=np.deg2rad(20)):
    """Mean power in two sectors of the annulus kmin < |k| < kmax: around
    the k1 axis and around the k2 axis."""
    k1, k2 = kgrid(n)
    kk = np.hypot(k1, k2)
    ang = np.arctan2(k2, k1)
    ann = (kk > kmin) & (kk < kmax) & inband(n)
    s1 = ann & (np.abs(ang) < half_angle)
    s2 = ann & (np.abs(np.abs(ang) - np.pi / 2) < half_angle)
    return P[s1].mean(), P[s2].mean()


def test_filler_removes_spin2_of_reconvolved_noise():
    """Monte Carlo through ImageQ: with an anisotropic PSF the reconvolved
    noise has more power along one axis than the other; after adding the
    filler the two sectors carry equal power, and the total spectrum is
    h^2 S^d_sym."""
    n, var, nreal = 64, 0.4, 1500
    sxx, syy = 0.14, 0.09  # PSF ellipticity ~0.2
    psf = gauss_psf(n, sxx, syy)
    ns = anacal.renoise.make_noise_spectrum(n, SCALE, var)
    rk = anacal.renoise.make_renoise_kernels(ns, psf, KLIM, 0.1, 0.05)
    imq = anacal.image.ImageQ(
        nx=n, ny=n, scale=SCALE, sigma_arcsec=SIGMA_H, klim=KLIM
    )
    rng = np.random.default_rng(7)
    P0 = np.zeros((n, n // 2 + 1))
    P1 = np.zeros_like(P0)
    for _ in range(nreal):
        eps = rng.normal(0.0, np.sqrt(var), (n, n))
        eps2 = rng.normal(0.0, np.sqrt(var), (n, n))
        v0 = imq.prepare_qnumber_image(
            eps.astype(np.float32), psf, n // 2, n // 2
        )[0]
        img = eps + apply(rk.g, eps2)
        v1 = imq.prepare_qnumber_image(
            img.astype(np.float32), psf, n // 2, n // 2
        )[0]
        P0 += np.abs(np.fft.rfft2(v0)) ** 2 / n**2
        P1 += np.abs(np.fft.rfft2(v1)) ** 2 / n**2
    P0 /= nreal
    P1 /= nreal
    k1, k2 = kgrid(n)
    h2 = np.exp(-(k1**2 + k2**2) * SIGMA_H**2)
    # the measured spectra match the prediction where the power is significant
    sig = inband(n) & (h2 * rk.Sd_sym > 1e-2 * (h2 * rk.Sd_sym).max())
    np.testing.assert_allclose(P0[sig], (h2 * rk.Sd)[sig], rtol=0.12)
    np.testing.assert_allclose(P1[sig], (h2 * rk.Sd_sym)[sig], rtol=0.12)
    # spin-2 part: sector ratio
    a0, b0 = sector_power(P0, n, 1.5, 4.0)
    a1, b1 = sector_power(P1, n, 1.5, 4.0)
    pa, pb = sector_power(h2 * rk.Sd, n, 1.5, 4.0)
    assert abs(pa / pb - 1) > 0.10, "test PSF is not anisotropic enough"
    np.testing.assert_allclose(a0 / b0, pa / pb, rtol=0.03)
    # after the filler the spectrum is R-symmetric; the sector estimator
    # itself is not exactly symmetric on the half-plane grid (the k1 = 0
    # column holds both k2 signs), so compare with the symmetric prediction
    pa1, pb1 = sector_power(h2 * rk.Sd_sym, n, 1.5, 4.0)
    assert abs(pa1 / pb1 - 1) < 0.01
    np.testing.assert_allclose(a1 / b1, pa1 / pb1, rtol=0.02)
    # the filler's cost in this configuration
    q = (h2 * (rk.Sd_sym - rk.Sd))[inband(n)].sum() / (h2 * rk.Sd)[
        inband(n)
    ].sum()
    assert 0.05 < q < 0.4


def central_correlation(a, b, lmax, w0, w1):
    """<a(x) b(x + r)> averaged over the window x in [w0, w1)^2, for
    lags |r_j| <= lmax; returns (2 lmax + 1, 2 lmax + 1)."""
    out = np.empty((2 * lmax + 1, 2 * lmax + 1))
    aw = a[w0:w1, w0:w1]
    for dy in range(-lmax, lmax + 1):
        for dx in range(-lmax, lmax + 1):
            out[dy + lmax, dx + lmax] = np.mean(
                aw * b[w0 + dy : w1 + dy, w0 + dx : w1 + dx]
            )
    return out


@pytest.mark.parametrize("c", [0.1, 0.01])
def test_assigned_response_cancels_shear_dependence(c):
    """The condition of the draft, through the pipeline's own D.

    The total noise zeta = eps_s^h + gamma D eps_s^h + n^h + gamma ndot^h,
    with eps_s the filler-symmetrized image noise, must have a
    correlation function independent of gamma to first order.  Its
    derivative at gamma = 0 is A + B with

        A(r) = <eps_s^h(x) (D eps_s^h)(x + r)> + (x <-> x + r),
        B(r) = <n^h(x) ndot^h(x + r)> + (x <-> x + r),

    and the draft predicts A = -B = F^-1[S^h_sym a_sym] (eq. Adef / asym).
    A is the sign test of the whole construction against the pipeline's D
    (Gaussian term plus coordinate terms, which make it the noisiest
    estimate here); B is the assigned response.
    """
    n, var, nreal, lmax = 96, 0.4, 240, 8
    w0, w1 = 32, 64
    sxx, syy, sxy = 0.14, 0.09, 0.015
    psf = gauss_psf(n, sxx, syy, sxy)
    ns = anacal.renoise.make_noise_spectrum(n, SCALE, var)
    rk = anacal.renoise.make_renoise_kernels(ns, psf, KLIM, c, 0.05)
    imq = anacal.image.ImageQ(
        nx=n, ny=n, scale=SCALE, sigma_arcsec=SIGMA_H, klim=KLIM
    )
    rng = np.random.default_rng(11)

    def smooth(img):
        return imq.prepare_qnumber_image(
            img.astype(np.float32), psf, n // 2, n // 2
        )

    def sym_corr(a, b):
        return central_correlation(a, b, lmax, w0, w1) + central_correlation(
            b, a, lmax, w0, w1
        )

    A = np.zeros((2 * lmax + 1,) * 2)
    B = np.zeros_like(A)
    A2 = np.zeros_like(A)
    B2 = np.zeros_like(A)
    for _ in range(nreal):
        eps = rng.normal(0.0, np.sqrt(var), (n, n))
        eps2 = rng.normal(0.0, np.sqrt(var), (n, n))
        eps1 = rng.normal(0.0, np.sqrt(var), (n, n))
        qs = smooth(eps + apply(rk.g, eps2))  # symmetrized image noise
        v_s, d_s = qs[0], qs[1]  # eps_s^h and D eps_s^h
        v_n = smooth(apply(rk.w, eps1))[0]  # n^h
        v_dn = smooth(apply(rk.kappa1, eps1))[0]  # ndot^h
        a = sym_corr(v_s, d_s)
        b = sym_corr(v_n, v_dn)
        A += a
        B += b
        A2 += a * a
        B2 += b * b
    A /= nreal
    B /= nreal
    sA = np.sqrt(np.maximum(A2 / nreal - A**2, 0) / nreal)
    sB = np.sqrt(np.maximum(B2 / nreal - B**2, 0) / nreal)
    # prediction: the inverse transform of S^h_sym a_sym
    k1, k2 = kgrid(n)
    h2 = np.exp(-(k1**2 + k2**2) * SIGMA_H**2)
    xi = np.fft.irfft2(
        h2 * rk.Sd_sym * rk.a_sym1, s=(n, n)
    )  # lag (0, 0) at [0, 0]
    pred = np.empty_like(A)
    for dy in range(-lmax, lmax + 1):
        for dx in range(-lmax, lmax + 1):
            pred[dy + lmax, dx + lmax] = xi[dy % n, dx % n]
    norm = np.sqrt(np.sum(pred**2))
    assert norm > 0.05 * var, "the shear changes the noise spectrum measurably"
    l2 = lambda x: np.sqrt(np.sum(x**2)) / norm  # noqa: E731
    chi2 = lambda x, sig: np.mean((x / sig) ** 2)  # noqa: E731
    # (i) the assigned response reproduces -F^-1[S^h_sym a_sym]
    assert l2(B + pred) < 0.05, l2(B + pred)
    # (ii) the pipeline's D changes the correlation of the symmetrized
    # noise as predicted: same sign and shape, within the Monte Carlo error
    assert l2(A - pred) < 0.3, l2(A - pred)
    assert chi2(A - pred, sA) < 1.5, chi2(A - pred, sA)
    # (iii) the total change cancels within the Monte Carlo error
    assert l2(A + B) < 0.3, l2(A + B)
    assert chi2(A + B, np.sqrt(sA**2 + sB**2)) < 1.5, chi2(
        A + B, np.sqrt(sA**2 + sB**2)
    )
    # ... and it is NOT cancelled without the assigned response (B = 0)
    assert chi2(A, sA) > 10.0


# ---------------------------------------------------------------------------
# Option 2 inside the qnumber image (ImageQ.prepare_qnumber_image with
# renoise_kernels): the image already carries the filler; from the one
# pure-noise image come the added noise w * noise in the value slot and
# the assigned response kappa_j * noise in the g_j slots.  Option 1 only
# changes the variance bookkeeping (total_spectrum).
# ---------------------------------------------------------------------------


def make_imq(n):
    return anacal.image.ImageQ(
        nx=n, ny=n, scale=SCALE, sigma_arcsec=SIGMA_H, klim=KLIM
    )


def qimage(imq, img, psf, n, noise=None, rk=None):
    kw = {}
    if noise is not None:
        kw["noise_array"] = noise.astype(np.float32)
    if rk is not None:
        kw["renoise_kernels"] = rk
    return imq.prepare_qnumber_image(
        img.astype(np.float32), psf, n // 2, n // 2, **kw
    )


def test_spectral_mode_reduces_to_plain_pipeline_without_noise():
    """With a zero noise image the added noise and the assigned response
    vanish, so every slot equals the plain (no renoising) pipeline bit for
    bit; the kernels need a noise image and a matching grid."""
    n, var = 64, 0.4
    psf = gauss_psf(n, 0.14, 0.09, 0.015)
    sym = symmetrized_spectrum(n, SCALE, var, psf, KLIM)
    rk = anacal.renoise.make_renoise_kernels(sym, psf, KLIM, 0.1, 0.0, False)
    rng = np.random.default_rng(3)
    img = rng.normal(0.0, 1.0, (n, n))
    imq = make_imq(n)
    plain = qimage(imq, img, psf, n)
    spec = qimage(imq, img, psf, n, noise=np.zeros((n, n)), rk=rk)
    assert np.array_equal(plain, spec)
    with pytest.raises(RuntimeError):
        qimage(imq, img, psf, n, rk=rk)
    rk32 = anacal.renoise.make_renoise_kernels(
        symmetrized_spectrum(32, SCALE, var, psf, KLIM),
        psf,
        KLIM,
        0.1,
        0.0,
        False,
    )
    with pytest.raises(RuntimeError):
        qimage(imq, img, psf, n, noise=np.zeros((n, n)), rk=rk32)


def test_symmetrized_spectrum():
    """symmetrize_spectrum: S_s = S S^d_sym / S^d for the given PSF, exactly
    90-degree symmetric after deconvolution, with the shear derivatives of
    the chain rule (a_sym + (k d) ln |p|^2), the filler filter g, and the
    raw spectrum kept for the pure-noise image."""
    n, var = 64, 0.4
    psf = gauss_psf(n, 0.14, 0.09, 0.015)
    ns = anacal.renoise.make_noise_spectrum(n, SCALE, var)
    rk = anacal.renoise.make_renoise_kernels(ns, psf, KLIM, 0.1, 0.05)
    sym = anacal.renoise.symmetrize_spectrum(ns, psf, KLIM, 0.05)
    ok = inband(n) & (rk.Sd > 0)
    assert sym.symmetrized and not ns.symmetrized
    np.testing.assert_allclose(
        sym.S[ok], var * rk.Sd_sym[ok] / rk.Sd[ok], rtol=1e-12
    )
    np.testing.assert_allclose(sym.S_raw, ns.S)
    np.testing.assert_allclose(sym.g[ok], rk.g[ok])
    np.testing.assert_allclose(
        sym.dS1[ok], (sym.S * (rk.a_sym1 + rk.ap1))[ok], rtol=1e-12, atol=1e-14
    )
    # kernels on the symmetrized spectrum without further symmetrization
    # see a symmetric S^d and reproduce a_sym
    rk2 = anacal.renoise.make_renoise_kernels(sym, psf, KLIM, 0.1, 0.0, False)
    np.testing.assert_allclose(rk2.Sd[ok], rk.Sd_sym[ok], rtol=1e-12)
    np.testing.assert_allclose(rk2.Sd_sym[ok], rk2.Sd[ok])
    np.testing.assert_allclose(
        rk2.a_sym1[ok], rk.a_sym1[ok], rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        rk2.a_sym2[ok], rk.a_sym2[ok], rtol=1e-9, atol=1e-12
    )
    rot = lambda a: anacal.renoise.rotate90(
        np.ascontiguousarray(a)
    )  # noqa: E731
    np.testing.assert_allclose(rot(rk2.Sd)[ok], rk2.Sd[ok], rtol=1e-12)
    # added noise: w^2 S_raw = c S_s (|p|^2 cancels), kappa = -a w / 2c
    np.testing.assert_allclose(
        (rk2.w**2 * sym.S_raw)[ok], (0.1 * sym.S)[ok], rtol=1e-12
    )
    np.testing.assert_allclose(
        rk2.kappa1[ok], (-rk2.a_sym1 * rk2.w / 0.2)[ok], rtol=1e-12, atol=1e-14
    )
    # a pure-noise image drawn from the rotated correlation (xlens): the
    # white case is unchanged, a correlated one gets sqrt(S / S_R)
    rk3 = anacal.renoise.make_renoise_kernels(
        sym, psf, KLIM, 0.1, 0.0, False, True
    )
    np.testing.assert_allclose(rk3.w, rk2.w)
    with pytest.raises(ValueError):
        anacal.renoise.symmetrize_spectrum(sym, psf, KLIM, 0.05)


def test_spectral_mode_terms():
    """Each term of the spectral mode against the pipeline's own primitives:
        v  = plain(image).v  + plain(w * noise).v
        g1 = plain(image).g1 + plain(kappa1 * noise).v
        g2 = plain(image).g2 + plain(kappa2 * noise).v
        x1 = plain(image).x1,  x2 = plain(image).x2
    (``w * noise`` the numpy filter of the stamp: the pipeline's transform
    differs from numpy's by a phase that commutes with a real filter)."""
    n, var = 64, 0.4
    psf = gauss_psf(n, 0.14, 0.09, 0.015)
    sym = symmetrized_spectrum(n, SCALE, var, psf, KLIM)
    rk = anacal.renoise.make_renoise_kernels(sym, psf, KLIM, 0.1, 0.0, False)
    rng = np.random.default_rng(5)
    img = rng.normal(0.0, 1.0, (n, n))
    eps = rng.normal(0.0, np.sqrt(var), (n, n)).astype(np.float32)
    imq = make_imq(n)
    e64 = eps.astype(np.float64)
    expect = qimage(imq, img, psf, n)
    expect[0] += qimage(imq, apply(rk.w, e64), psf, n)[0]
    expect[1] += qimage(imq, apply(rk.kappa1, e64), psf, n)[0]
    expect[2] += qimage(imq, apply(rk.kappa2, e64), psf, n)[0]
    got = qimage(imq, img, psf, n, noise=eps, rk=rk)
    scale = np.abs(expect).max()
    np.testing.assert_allclose(got, expect, atol=2e-6 * scale, rtol=0)


def test_spectral_mode_noise_statistics():
    """Monte Carlo of pure noise through the spectral mode, on an image
    whose noise carries the spin-2 filler of the same PSF: the value slot
    has the symmetrized spectrum (1 + c) h^2 S^d_s (variance_factor gives
    its variance relative to the plain smoothed white noise, the sector
    ratio is 1), and the first-order shear derivative of its correlation
    function, <v (g1 slot)> symmetrized, vanishes -- while it does not for
    the plain pipeline on the raw image noise."""
    n, var, c, nreal, lmax = 96, 0.4, 0.1, 300, 6
    w0, w1 = 32, 64
    psf = gauss_psf(n, 0.14, 0.09, 0.015)
    sym = symmetrized_spectrum(n, SCALE, var, psf, KLIM)
    rk = anacal.renoise.make_renoise_kernels(sym, psf, KLIM, c, 0.0, False)
    imq = make_imq(n)
    rng = np.random.default_rng(17)

    def sym_corr(a, b):
        return central_correlation(a, b, lmax, w0, w1) + central_correlation(
            b, a, lmax, w0, w1
        )

    P = np.zeros((n, n // 2 + 1))
    P_plain = np.zeros_like(P)
    C = np.zeros((2 * lmax + 1,) * 2)
    C2 = np.zeros_like(C)
    Cp = np.zeros_like(C)
    Cp2 = np.zeros_like(C)
    for _ in range(nreal):
        eps = rng.normal(0.0, np.sqrt(var), (n, n))
        eps2 = rng.normal(0.0, np.sqrt(var), (n, n))
        eps1 = rng.normal(0.0, np.sqrt(var), (n, n))
        eps_s = eps + apply(sym.g, eps2)  # the image noise with the filler
        q = qimage(imq, eps_s, psf, n, noise=eps1, rk=rk)
        qp = qimage(imq, eps, psf, n)
        P += np.abs(np.fft.rfft2(q[0])) ** 2 / n**2
        P_plain += np.abs(np.fft.rfft2(qp[0])) ** 2 / n**2
        cc = sym_corr(q[0], q[1])
        cp = sym_corr(qp[0], qp[1])
        C += cc
        C2 += cc * cc
        Cp += cp
        Cp2 += cp * cp
    P /= nreal
    P_plain /= nreal
    C /= nreal
    Cp /= nreal
    sC = np.sqrt(np.maximum(C2 / nreal - C**2, 0) / nreal)
    sCp = np.sqrt(np.maximum(Cp2 / nreal - Cp**2, 0) / nreal)

    ratio = P[inband(n)].sum() / P_plain[inband(n)].sum()
    pred = anacal.renoise.variance_factor(rk, SIGMA_H)
    assert 1.05 < pred < 1.6
    np.testing.assert_allclose(ratio, pred, rtol=0.03)
    k1, k2 = kgrid(n)
    h2 = np.exp(-(k1**2 + k2**2) * SIGMA_H**2)
    a1, b1 = sector_power(P, n, 1.5, 4.0)
    pa, pb = sector_power(h2 * rk.Sd_sym, n, 1.5, 4.0)
    np.testing.assert_allclose(a1 / b1, pa / pb, rtol=0.03)
    a0, b0 = sector_power(P_plain, n, 1.5, 4.0)
    assert abs(a0 / b0 - 1) > 0.1
    chi2 = lambda x, s: np.mean((x / s) ** 2)  # noqa: E731
    assert chi2(C, sC) < 1.5, chi2(C, sC)
    assert chi2(Cp, sCp) > 10.0, chi2(Cp, sCp)


def test_rotate_mode_total_spectrum():
    """Option 1 through the spectrum: for white noise the rotated copy
    doubles every Gaussian-window variance exactly (the window is
    isotropic, so the rotated deconvolved spectrum sums to the same), and
    a Setup refuses a spectrum of the wrong kind."""
    n, var = 64, 0.3
    psf = gauss_psf(n, 0.14, 0.09, 0.015)
    ns = anacal.renoise.make_noise_spectrum(n, SCALE, var, None)
    rk = anacal.renoise.make_renoise_kernels(
        ns, psf, KLIM, 0.1, 0.0, False, True, mode="rotate"
    )
    assert rk.mode == "rotate"
    np.testing.assert_allclose(
        anacal.renoise.variance_factor(rk, SIGMA_H), 2.0, rtol=1e-12
    )
    ref = anacal.task.gaussian_flux_variance(
        psf_array=psf, sigma_kernel=0.0, sigma_smooth=SIGMA_H,
        pixel_scale=SCALE, klim=KLIM,
    )
    np.testing.assert_allclose(
        anacal.renoise.gaussian_flux_variance(rk, 0.0, SIGMA_H),
        2.0 * var * ref, rtol=1e-10,
    )
    total = anacal.renoise.total_spectrum(rk)
    np.testing.assert_allclose(total, np.asarray(rk.Sd) + np.asarray(rk.Sd_rot))
    # the mode check of the Setup
    sym = anacal.renoise.symmetrize_spectrum(ns, psf, KLIM, 0.05)
    with pytest.raises(RuntimeError, match="raw noise spectrum"):
        anacal.renoise.make_cell_kernels(
            anacal.renoise.Setup(spectra=[sym], mode="rotate"), psf, KLIM
        )
    with pytest.raises(RuntimeError, match="symmetrized"):
        anacal.renoise.make_cell_kernels(
            anacal.renoise.Setup(spectra=[ns], mode="spectral"), psf, KLIM
        )
