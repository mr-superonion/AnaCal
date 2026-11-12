"""Spectral renoising (Option 2) in the forced FPFS measurement
(anacal.fpfs.process_image -> ForceTask with renoise_mode="spectral").

All three kernels are applied in Fourier space per stamp: the spin-2
filler rho * noise2 (from the second, independent realization eps'') joins
the image modes for the value AND the analytic shear response, the added
noise eta0 * noise joins the value, and kappa_j * noise the shear response.
The measurement is linear in the image, so every term can be checked
exactly against process_image itself run on the separate pieces; the
covariance of each mode is checked against the scatter over noise
realizations.  The agreement with the Task is tested in
test_task_force_consistency.py.
"""

import anacal
import numpy as np
import pytest

from ..fixtures import (
    COSMOS_BUFF,
    COSMOS_MAG_ZERO,
    COSMOS_NGRID,
    COSMOS_NSTAMP,
    cosmos_centres,
    cosmos_scene,
)

SCALE = 0.2
MAG_ZERO = COSMOS_MAG_ZERO
NSTAMP, NGRID, BUFF = COSMOS_NSTAMP, COSMOS_NGRID, COSMOS_BUFF
C, DELTA = 0.1, 0.05
SIGMA = 0.52


def scene(seed=1, noise_std=0.37):
    gal, psf = cosmos_scene()
    var = noise_std**2
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, noise_std, gal.shape)
    eps2 = rng.normal(0.0, noise_std, gal.shape)
    noise = rng.normal(0.0, noise_std, gal.shape)
    return gal, eps, eps2, noise, psf, var, cosmos_centres()


def fpfs_klim(cfg, psf):
    ftask = anacal.fpfs.FpfsTask(
        npix=cfg.npix,
        pixel_scale=SCALE,
        sigma_shapelets=SIGMA,
        psf_array=psf,
        kmax=cfg.kmax,
        kmax_thres=cfg.kmax_thres,
    )
    return ftask.kmax / SCALE, ftask


def run(cfg, img, psf, det, var, noise=None, **kw):
    return anacal.fpfs.process_image(
        fpfs_config=cfg,
        mag_zero=MAG_ZERO,
        gal_array=np.asarray(img, dtype=np.float32),
        psf_array=psf,
        pixel_scale=SCALE,
        noise_variance=var,
        noise_array=(
            None if noise is None else np.asarray(noise, dtype=np.float32)
        ),
        detection=det,
        **kw,
    )


def apply(filt, img):
    """Filter a full image with a stamp-grid kernel: tile-wise, since the
    forced measurement transforms each source's 64 x 64 cutout."""
    out = np.zeros_like(img)
    n = NGRID
    for i in range(NSTAMP):
        for j in range(NSTAMP):
            sl = (
                slice(BUFF + i * n, BUFF + (i + 1) * n),
                slice(BUFF + j * n, BUFF + (j + 1) * n),
            )
            out[sl] = np.fft.irfft2(np.fft.rfft2(img[sl]) * filt, s=(n, n))
    return out


@pytest.mark.parametrize("c", [C, 0.01])
def test_spectral_forced_measurement_terms(c):
    gal, eps, eps2, noise, psf, var, det = scene()
    # shape_order=4: the check below includes the q response (the other
    # spectral tests run at the default order 2, the 9-mode path)
    cfg = anacal.fpfs.FpfsConfig(sigma_shapelets1=SIGMA, shape_order=4)
    klim, _ = fpfs_klim(cfg, psf)
    c0 = cfg.c0 * 10 ** ((MAG_ZERO - anacal.fpfs.THRESHOLD_REF_MAG_ZERO) / 2.5)
    img = gal + eps
    ns = anacal.renoise.make_noise_spectrum(NGRID, SCALE, var)
    rk = anacal.renoise.make_renoise_kernels(ns, psf, klim, c, DELTA, True)

    got = run(
        cfg,
        img,
        psf,
        det,
        var,
        noise=noise,
        noise2_array=np.asarray(eps2, dtype=np.float32),
        renoise_mode="spectral",
        renoise_c=c,
        renoise_delta=DELTA,
        noise_rotated=False,
    )
    base = run(cfg, img, psf, det, var)
    # the filler: measured like image, so it contributes to the values
    # AND (through the analytic response) to the shear responses
    cphi = run(cfg, apply(rk.rho, eps2), psf, det, var)
    cn = run(cfg, apply(rk.eta0, noise), psf, det, var)
    ck1 = run(cfg, apply(rk.tau1, noise), psf, det, var)
    ck2 = run(cfg, apply(rk.tau2, noise), psf, det, var)

    k = "fpfs1"
    modes = ("m00", "m20", "m22c", "m22s", "m42c", "m42s")
    val = {
        m: base[f"{k}_{m}"] + cphi[f"{k}_{m}"] + cn[f"{k}_{m}"]
        for m in modes
    }
    dg = {}
    for m in modes:
        dg[f"d{m}_dg1"] = (
            base[f"{k}_d{m}_dg1"] + cphi[f"{k}_d{m}_dg1"] + ck1[f"{k}_{m}"]
        )
        dg[f"d{m}_dg2"] = (
            base[f"{k}_d{m}_dg2"] + cphi[f"{k}_d{m}_dg2"] + ck2[f"{k}_{m}"]
        )
    denom = val["m00"] + c0

    def close(a, b):
        # float32 input rounding: the pieces are cast separately
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-6 * np.abs(b).max())

    for m in modes:
        close(got[f"{k}_{m}"], val[m])
        close(got[f"{k}_d{m}_dg1"], dg[f"d{m}_dg1"])
        close(got[f"{k}_d{m}_dg2"], dg[f"d{m}_dg2"])
    close(got[f"{k}_e1"], val["m22c"] / denom)
    close(
        got[f"{k}_de1_dg1"],
        dg["dm22c_dg1"] / denom - dg["dm00_dg1"] * val["m22c"] / denom**2,
    )
    close(
        got[f"{k}_de2_dg2"],
        dg["dm22s_dg2"] / denom - dg["dm00_dg2"] * val["m22s"] / denom**2,
    )
    close(
        got[f"{k}_dq1_dg1"],
        dg["dm42c_dg1"] / denom - dg["dm00_dg1"] * val["m42c"] / denom**2,
    )
    # the flux error is the (1 + c) S^d_s covariance, below the rotated one
    rot = run(cfg, img, psf, det, var, noise=noise)
    r = got[f"flux_{k}_err"][0] / rot[f"flux_{k}_err"][0]
    _, ftask = fpfs_klim(cfg, psf)
    ftask.prepare_covariance(variance=var, renoise_kernels=rk)
    s_spec = ftask.std_m00
    ftask.prepare_covariance(variance=var)
    s_rot = ftask.std_m00
    np.testing.assert_allclose(r, s_spec / s_rot, rtol=1e-10)
    assert 0.65 < r < 0.85, r


def test_spectral_forced_measurement_scatter():
    """The measured scatter of m00 over noise realizations matches the
    covariance of each mode: no correction, rotated noise (x2) and
    spectral ((1 + c) S^d_s, the filler applied in Fourier space)."""
    cfg = anacal.fpfs.FpfsConfig(sigma_shapelets1=SIGMA)
    gal, _, _, _, psf, var, det = scene()
    rng = np.random.default_rng(5)
    sd = np.sqrt(var)
    spec_kw = dict(
        renoise_mode="spectral",
        renoise_c=C,
        renoise_delta=DELTA,
        noise_rotated=False,
    )
    samples = {"rotate": [], "spectral": [], "none": []}
    for _ in range(12):
        eps = rng.normal(0.0, sd, gal.shape)
        eps2 = rng.normal(0.0, sd, gal.shape).astype(np.float32)
        noise = rng.normal(0.0, sd, gal.shape)
        img = gal + eps
        samples["none"].append(run(cfg, img, psf, det, var)["fpfs1_m00"])
        samples["rotate"].append(
            run(cfg, img, psf, det, var, noise=noise)["fpfs1_m00"]
        )
        samples["spectral"].append(
            run(
                cfg, img, psf, det, var,
                noise=noise, noise2_array=eps2, **spec_kw,
            )["fpfs1_m00"]
        )
    truth = run(cfg, gal, psf, det, var)["fpfs1_m00"]
    std = {m: np.std(np.array(v) - truth) for m, v in samples.items()}
    fac = anacal.fpfs.m00_to_flux(1.0, SIGMA)
    err_rot = (
        run(cfg, img, psf, det, var, noise=noise)["flux_fpfs1_err"][0] / fac
    )
    err_spec = (
        run(
            cfg, img, psf, det, var,
            noise=noise, noise2_array=eps2, **spec_kw,
        )["flux_fpfs1_err"][0]
        / fac
    )
    # 1200 samples: 2% on a standard deviation.  Without a noise image the
    # catalog's flux error still carries the rotated-noise doubling.
    np.testing.assert_allclose(std["none"], err_rot / np.sqrt(2.0), rtol=0.08)
    np.testing.assert_allclose(std["rotate"], err_rot, rtol=0.08)
    np.testing.assert_allclose(std["spectral"], err_spec, rtol=0.08)
    assert std["none"] < std["spectral"] < std["rotate"]


def test_spectral_forced_measurement_errors():
    gal, eps, eps2, noise, psf, var, det = scene()
    cfg = anacal.fpfs.FpfsConfig(sigma_shapelets1=SIGMA)
    with pytest.raises(ValueError):
        run(cfg, gal + eps, psf, det, var, noise=noise, renoise_mode="other")
    # spectral with a noise image needs the second realization eps''
    with pytest.raises(ValueError):
        run(
            cfg, gal + eps, psf, det, var, noise=noise,
            renoise_mode="spectral",
        )
    # without a noise image the mode is irrelevant: plain measurement
    # (per column: the q family is NaN at the default shape_order=2)
    a = run(cfg, gal + eps, psf, det, var, renoise_mode="spectral")
    b = run(cfg, gal + eps, psf, det, var)
    for name in a.dtype.names:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
    # only the 2nd- and 4th-order estimators exist
    cfg3 = anacal.fpfs.FpfsConfig(sigma_shapelets1=SIGMA, shape_order=3)
    with pytest.raises(ValueError, match="shape_order"):
        run(cfg3, gal + eps, psf, det, var)
