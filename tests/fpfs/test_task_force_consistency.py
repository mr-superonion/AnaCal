# Consistency between the detection / measurement Task (anacal.task.Task,
# FPFS moments at sigma_arcsec) and the forced FPFS measurement
# (anacal.fpfs.process_image at sigma_shapelets = sigma_arcsec sqrt 2):
# on a 256 x 256 image tiled with 4 x 4 stamps of 64 pixels, one galaxy
# per stamp at the stamp centre, with the source centres given as input.
# The Task runs with one 64-pixel cell per stamp, so every cell is the
# forced measurement's source cutout: same pixels, same periodic domain.
#
# Checked, per source, for a noiseless image, a noisy image without
# correction, the rotated-noise correction on white ("rotate") and on
# correlated ("rotate_corr") noise, and the spectral renoising
# ("spectral"):
#   values     fpfs_m0 = m00, fpfs_m2 = m00 + m20, fpfs_e1 = e1, fpfs_e2 = e2
#   responses  d fpfs_m0 / dg, d fpfs_m2 / dg, d fpfs_e / dg  (both components)
#   flux       flux_gauss0 = flux_fpfs1 and flux_gauss0_err = flux_fpfs1_err
# The forced measurement uses the Task's Fourier cutoff by default
# (FpfsConfig.kmax = anacal.task.KLIM_PIX); the Task's real-space moment
# window (7 sigma, fitting.h sigma2_lim) is what limits the agreement.
# "rotate_corr" is Option 1 on correlated noise, with the noise spectrum
# of the patch given to both codes.
import anacal
import numpy as np
import pytest

from ..fixtures import load, make_filler, symmetrized_spectrum

SCALE = 0.2
SIGMA_ARCSEC = 0.38
STAMP = 64
NTILE = 4
NPIX = STAMP * NTILE
KLIM = 3.1 / SCALE
NOISE_STD = 0.2
RENOISE_C = 0.1
RENOISE_DELTA = 0.05
MODES = ["noiseless", "none", "rotate", "rotate_corr", "spectral"]
# an asymmetric kernel: correlated noise without any symmetry
CORR_KERNEL = np.array(
    [[0.0, 0.15, 0.0], [0.35, 1.0, 0.5], [0.0, 0.1, 0.0]]
)


def _correlated_noise(rng, shape, std, kernel):
    """White noise convolved (periodically) with ``kernel``, scaled to
    ``std`` per pixel, and its correlation function on a 9 x 9 window
    centred on pixel (4, 4) with C(0) = std**2."""
    n = shape[0]
    k = np.zeros(shape)
    kh = kernel.shape[0] // 2
    k[: kernel.shape[0], : kernel.shape[1]] = kernel
    k = np.roll(k, (-kh, -kh), axis=(0, 1))
    kf = np.fft.rfft2(k)
    white = rng.normal(0.0, 1.0, shape)
    field = np.fft.irfft2(np.fft.rfft2(white) * kf, s=shape)
    corr = np.fft.irfft2(np.abs(kf) ** 2, s=shape)
    norm = corr[0, 0]
    corr = np.fft.fftshift(corr) / norm * std**2
    c = n // 2
    return field * std / np.sqrt(norm), corr[c - 4:c + 5, c - 4:c + 5]


def _rotate_flux_variance(psf, corr, var, sigma):
    """Numpy prediction of the Gaussian flux variance under Option 1 with
    correlated noise: sum of the aperture window over S^d(k) + S^d(R k),
    S^d = S / |p|^2 with the deconvolution floor and cutoff of anacal."""
    n = STAMP
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=SCALE)
    k2, k1 = np.meshgrid(k, k)  # k2 along rows (axis 0), k1 along columns
    p = np.fft.fft2(np.fft.ifftshift(psf))
    pabs = np.maximum(np.abs(p), 1e-5 * p[0, 0].real)
    cc = np.zeros((n, n))
    h = corr.shape[0] // 2
    cc[n // 2 - h:n // 2 + h + 1, n // 2 - h:n // 2 + h + 1] = corr
    if corr is None:
        cc[n // 2, n // 2] = var
    s = np.fft.fft2(np.fft.ifftshift(cc)).real
    band = (k1**2 + k2**2) < KLIM**2
    sd = np.where(band, s / pabs**2, 0.0)
    # S^d(R k) on the FFT grid: (k1, k2) -> (-k2, k1), index by index (an
    # even grid has no centre pixel, so rot90 of the shifted array would
    # be off by one sample)
    jj, ii = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    sd_rot = sd[ii, (-jj) % n]
    w2 = np.exp(-(k1**2 + k2**2) * 2.0 * sigma**2)
    ff = 4.0 * np.pi * (sigma / SCALE) ** 2
    return np.sum(w2 * (sd + sd_rot)) * ff**2 / n**2


def _render_galaxies(psf, rng):
    """16 PSF-convolved exponential galaxies (varied size, shape and
    flux), each at the centre pixel (32 + 64 i, 32 + 64 j) of its stamp."""
    y, x = (np.mgrid[0:STAMP, 0:STAMP] - STAMP // 2) * SCALE
    pk = np.fft.rfft2(np.fft.ifftshift(psf))
    img = np.zeros((NPIX, NPIX))
    centres = []
    for j in range(NTILE):
        for i in range(NTILE):
            hlr = rng.uniform(0.18, 0.40)
            e = rng.uniform(0.0, 0.35)
            theta = rng.uniform(0.0, np.pi)
            flux = rng.uniform(60.0, 300.0)
            xr = x * np.cos(theta) + y * np.sin(theta)
            yr = -x * np.sin(theta) + y * np.cos(theta)
            r = np.sqrt(xr**2 / (1.0 + e) + yr**2 / (1.0 - e))
            gal = np.exp(-1.678 * r / hlr)
            gal = gal / gal.sum() * flux
            gal = np.fft.irfft2(np.fft.rfft2(gal) * pk, s=(STAMP, STAMP))
            img[j * STAMP:(j + 1) * STAMP, i * STAMP:(i + 1) * STAMP] = gal
            centres.append((j * STAMP + STAMP // 2, i * STAMP + STAMP // 2))
    return img, np.array(centres, dtype=np.float64)


def _task_detection(centres):
    """Task input catalog with the source centres (arcsec)."""
    x1 = centres[:, 1] * SCALE
    x2 = centres[:, 0] * SCALE
    det = anacal.table.make_catalog_empty(x1, x2)
    det["x1"] = x1
    det["x2"] = x2
    det["x1_det"] = x1
    det["x2_det"] = x2
    return det


def _run(mode, seed=7):
    fix = load("ngmix_fpfs")
    psf = np.asarray(fix["psf"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    gal, centres = _render_galaxies(psf, rng)
    var = NOISE_STD**2
    img = gal.copy()
    noise = None
    corr = None
    renoise = None
    fp_kw = {}
    if mode == "rotate_corr":
        # correlated image noise; the pure noise drawn from the rotated
        # correlation (xlens convention, noise_rotated=True)
        eps, corr = _correlated_noise(rng, gal.shape, NOISE_STD, CORR_KERNEL)
        img = img + eps
        noise = _correlated_noise(
            rng, gal.shape, NOISE_STD, np.rot90(CORR_KERNEL)
        )[0].astype(np.float32)
    elif mode != "noiseless":
        img = img + rng.normal(0.0, NOISE_STD, gal.shape)
    if mode in ("rotate", "spectral"):
        noise = rng.normal(0.0, NOISE_STD, gal.shape).astype(np.float32)
    if mode in ("rotate", "rotate_corr"):
        # Option 1 with the noise spectrum of the patch: the variances
        # are summed over S^d + S^d(R k)
        raw = anacal.renoise.make_noise_spectrum(STAMP, SCALE, var, corr)
        renoise = anacal.renoise.Setup(
            spectra=[raw], noise_rotated=True, mode="rotate"
        )
        fp_kw = dict(
            renoise_mode="rotate", noise_corr=corr, noise_rotated=True
        )
    if mode == "spectral":
        eps2 = rng.normal(0.0, NOISE_STD, gal.shape)
        img = img + make_filler(eps2, SCALE, var, psf, KLIM, RENOISE_DELTA)
        sym = symmetrized_spectrum(STAMP, SCALE, var, psf, KLIM, RENOISE_DELTA)
        renoise = anacal.renoise.Setup(
            c=RENOISE_C, spectra=[sym], noise_rotated=False, mode="spectral"
        )
        fp_kw = dict(
            renoise_mode="spectral",
            renoise_c=RENOISE_C,
            renoise_delta=RENOISE_DELTA,
            noise_rotated=False,
            filler_psf=psf,
        )
    img = img.astype(np.float32)

    task = anacal.task.Task(
        scale=SCALE,
        sigma_arcsec=SIGMA_ARCSEC,
        snr_peak_min=5.0,
        omega_f=0.218,
        omega_v=0.011,
        stamp_size=32,
        num_epochs=20,
        force_center=True,
        fpfs_c0=0.0,
    )
    cells = anacal.geometry.get_cell_list(
        img_nx=NPIX, img_ny=NPIX, cell_nx=STAMP, cell_ny=STAMP,
        cell_overlap=0, scale=SCALE,
    )
    assert len(cells) == NTILE**2
    assert all((c.nx == STAMP) and (c.ny == STAMP) for c in cells)
    cat = task.process_image(
        img,
        psf,
        variance=var,
        cell_list=cells,
        detection=_task_detection(centres),
        noise_array=noise,
        renoise=renoise,
    )
    assert len(cat) == NTILE**2
    cat = cat[np.lexsort((cat["x1_det"], cat["x2_det"]))]

    # FpfsConfig.kmax defaults to the Task's Fourier cutoff
    cfg = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=SIGMA_ARCSEC * np.sqrt(2.0), c0=0.0
    )
    assert cfg.kmax == anacal.task.KLIM_PIX
    det = np.zeros(NTILE**2, dtype=[("y", "f8"), ("x", "f8")])
    det["y"] = cat["x2"] / SCALE
    det["x"] = cat["x1"] / SCALE
    fp = anacal.fpfs.process_image(
        fpfs_config=cfg,
        mag_zero=anacal.fpfs.THRESHOLD_REF_MAG_ZERO,
        gal_array=img,
        psf_array=psf,
        pixel_scale=SCALE,
        noise_variance=var,
        noise_array=noise,
        detection=det,
        **fp_kw,
    )
    assert len(fp) == NTILE**2
    return cat, fp, (psf, corr)


@pytest.mark.parametrize("mode", MODES)
def test_task_force_consistency(mode):
    """Same image, same pure-noise image, same grid and the same Fourier
    cutoff: the Task's FPFS moments and their shear responses, computed
    as real-space sums over a 7 sigma window, equal those of the forced
    measurement, computed in Fourier space, to double precision roundoff
    for the noiseless image, the uncorrected noisy image, the rotated
    noise and the spectral renoising alike (S/N 25 - 135 here)."""
    cat, fp, (psf, corr) = _run(mode)
    k = "fpfs1"
    # FPFS values
    np.testing.assert_allclose(cat["fpfs_m0"], fp[f"{k}_m00"], rtol=1e-11)
    np.testing.assert_allclose(
        cat["fpfs_m2"], fp[f"{k}_m00"] + fp[f"{k}_m20"], rtol=1e-8
    )
    np.testing.assert_allclose(cat["fpfs_e1"], fp[f"{k}_e1"], atol=1e-9)
    np.testing.assert_allclose(cat["fpfs_e2"], fp[f"{k}_e2"], atol=1e-9)
    # shear responses (where the noise-bias correction enters)
    for a, b in (("fpfs_dm0_dg1", "dm00_dg1"), ("fpfs_dm0_dg2", "dm00_dg2")):
        np.testing.assert_allclose(cat[a], fp[f"{k}_{b}"], rtol=1e-7, atol=1e-8)
    for a, b in (
        ("fpfs_de1_dg1", "de1_dg1"),
        ("fpfs_de2_dg2", "de2_dg2"),
        ("fpfs_de1_dg2", "de1_dg2"),
        ("fpfs_de2_dg1", "de2_dg1"),
    ):
        np.testing.assert_allclose(cat[a], fp[f"{k}_{b}"], rtol=1e-6, atol=1e-8)
    for a, b, c in (
        ("fpfs_dm2_dg1", "dm00_dg1", "dm20_dg1"),
        ("fpfs_dm2_dg2", "dm00_dg2", "dm20_dg2"),
    ):
        np.testing.assert_allclose(
            cat[a], fp[f"{k}_{b}"] + fp[f"{k}_{c}"], rtol=1e-5, atol=1e-6
        )
    # Gaussian flux: the Task's aperture flux at sigma_arcsec is the
    # forced m00 converted to flux, and its error is the forced flux error
    # in every mode (both summed over the actual noise spectrum: image
    # noise alone, plus the rotated copy, or (1 + c) times the symmetrized
    # spectrum)
    np.testing.assert_allclose(
        cat["flux_gauss0"], fp[f"flux_{k}"], rtol=1e-10
    )
    np.testing.assert_allclose(
        cat["flux_gauss0_err"], fp[f"flux_{k}_err"], rtol=1e-6
    )
    if mode == "rotate_corr":
        pred = np.sqrt(
            _rotate_flux_variance(psf, corr, NOISE_STD**2, SIGMA_ARCSEC)
        )
        np.testing.assert_allclose(fp[f"flux_{k}_err"], pred, rtol=1e-5)
    if mode == "rotate":
        # white noise: the rotated copy doubles the variance exactly
        cat0, fp0, _ = _run("none")
        np.testing.assert_allclose(
            fp[f"flux_{k}_err"], np.sqrt(2.0) * fp0[f"flux_{k}_err"], rtol=1e-10
        )


def test_corrections_change_response():
    """The two noise corrections change the responses of the same noisy
    image measurably (so the agreement above is not trivial), and the
    spectral renoising adds much less noise than the rotated copy."""
    cat0, _, _ = _run("none")
    cat1, fp1, _ = _run("rotate")
    cat2, fp2, _ = _run("spectral")
    d1 = np.abs(cat1["fpfs_dm0_dg1"] - cat0["fpfs_dm0_dg1"])
    d2 = np.abs(cat2["fpfs_dm0_dg1"] - cat0["fpfs_dm0_dg1"])
    assert np.max(d1) > 1e-2
    assert np.max(d2) > 1e-2
    assert np.all(fp2["flux_fpfs1_err"] < 0.85 * fp1["flux_fpfs1_err"])
