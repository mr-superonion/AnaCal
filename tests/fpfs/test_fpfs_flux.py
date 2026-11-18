"""The m00 -> flux conversion and its ``sigma_psf`` split.

m00 is the PSF-deconvolved image smoothed with a unit Gaussian of width
sigma_shapelets, read at the centre.  Split into a reconvolution PSF of
width sigma_psf and a measurement window sigma_w (sigma_w^2 =
sigma_shapelets^2 - sigma_psf^2), the Gaussian-aperture flux of the
reconvolved image normalised as the removed GaussFit::measure_flux did
(sum(w d) 2 pi sigma_w^2 / sum(w^2)) is m00 * 4 pi sigma_w^2.

Checks: the default (sigma_psf <= 0) is unchanged and equals the
symmetric split sigma_psf = sigma_shapelets / sqrt(2); a point source's
flux follows the analytic aperture fraction; the removed numpy formula
is reproduced on a reconvolved image for the flux_gauss0 and
flux_gauss2 windows; the adaptive kernel's flux response carries the
window's growth; the error paths.
"""

import anacal
import numpy as np
import pytest

from ..fixtures import cosmos_centres, cosmos_scene

SCALE = 0.2
NPIX = 64
FLUX = 1000.0


def _delta():
    psf = np.zeros((NPIX, NPIX))
    psf[NPIX // 2, NPIX // 2] = 1.0
    return psf


def _det_centre():
    det = np.zeros(1, dtype=[("y", "f8"), ("x", "f8")])
    det["x"] = NPIX // 2
    det["y"] = NPIX // 2
    return det


def _gaussian_image(sigma_x, sigma_y, rho=0.0):
    """A Gaussian galaxy of FLUX counts sampled at the pixel centres,
    centred on pixel (NPIX // 2, NPIX // 2) like the AnaCal stamps."""
    x = (np.arange(NPIX) - NPIX // 2) * SCALE
    xx, yy = np.meshgrid(x, x)
    q = (
        (xx / sigma_x) ** 2 + (yy / sigma_y) ** 2
        - 2.0 * rho * xx * yy / (sigma_x * sigma_y)
    ) / (1.0 - rho**2)
    img = np.exp(-0.5 * q)
    return img * (FLUX / img.sum())


def _process(img, psf, det, **cfg):
    return anacal.fpfs.process_image(
        fpfs_config=anacal.fpfs.FpfsConfig(**cfg),
        mag_zero=30.0,
        gal_array=img,
        psf_array=psf,
        pixel_scale=SCALE,
        noise_variance=1e-6,
        detection=det,
    )


def _reconvolve(img, sigma_psf):
    """The image (a delta PSF, so already deconvolved) smoothed with a
    unit Gaussian of width sigma_psf, counts per pixel."""
    kx = np.fft.fftfreq(NPIX, d=SCALE) * 2.0 * np.pi
    k2 = kx[None, :] ** 2 + kx[:, None] ** 2
    smooth = np.exp(-0.5 * k2 * sigma_psf**2)
    return np.fft.ifft2(np.fft.fft2(img) * smooth).real


def _old_gaussian_flux(data, sigma_meas2):
    """GaussFit::measure_flux of AnaCal before d8b4ac1 (flux_gauss0/2):
    sum(w d) * 2 pi sigma_meas^2 / sum(w^2) / scale^2 with the window
    w = exp(-r^2 / 2 sigma_meas^2) about the source, r in arcsec."""
    x = (np.arange(NPIX) - NPIX // 2) * SCALE
    xx, yy = np.meshgrid(x, x)
    w = np.exp(-0.5 * (xx**2 + yy**2) / sigma_meas2)
    m0 = np.sum(w * data)
    norm = np.sum(w * w)
    return m0 * (2.0 * np.pi * sigma_meas2) / norm / SCALE**2


FLUX_FAMILY = (
    "flux_fpfs1", "dflux_fpfs1_dg1", "dflux_fpfs1_dg2", "flux_fpfs1_err",
    "s2n_fpfs1", "ds2n_fpfs1_dg1", "ds2n_fpfs1_dg2",
)


def test_symmetric_split_is_the_default():
    """sigma_psf = sigma_shapelets / sqrt(2) reproduces the default flux
    (4 pi (s^2 - s^2 / 2) = 2 pi s^2) to rounding, and leaves every
    other column untouched."""
    sigma = 0.53
    m00 = np.array([0.3, 1.7, -0.2])
    np.testing.assert_allclose(
        anacal.fpfs.m00_to_flux(m00, sigma, sigma / np.sqrt(2.0)),
        anacal.fpfs.m00_to_flux(m00, sigma),
        rtol=1e-14,
    )
    assert anacal.fpfs.m00_to_flux(1.0, sigma) == 2.0 * np.pi * sigma**2

    img = _gaussian_image(0.5, 0.35, 0.2)
    ref = _process(img, _delta(), _det_centre(), sigma_shapelets1=sigma)
    sym = _process(
        img, _delta(), _det_centre(),
        sigma_shapelets1=sigma, sigma_psf=sigma / np.sqrt(2.0),
    )
    for name in ref.dtype.names:
        if name in FLUX_FAMILY:
            np.testing.assert_allclose(sym[name], ref[name], rtol=1e-14)
        else:
            np.testing.assert_array_equal(sym[name], ref[name])


@pytest.mark.parametrize("sigma_psf", [-1.0, 0.3, 0.45, 0.55])
def test_point_source_flux(sigma_psf):
    """A point source of flux F: the aperture of width sigma_w on the
    reconvolved PSF holds F sigma_w^2 / (sigma_psf^2 + sigma_w^2), and
    the normalisation doubles it: 2 F sigma_w^2 / sigma_shapelets^2 (F
    itself for the symmetric split)."""
    sigma = 0.6
    img = np.zeros((NPIX, NPIX))
    img[NPIX // 2, NPIX // 2] = FLUX
    cat = _process(
        img, _delta(), _det_centre(),
        sigma_shapelets1=sigma, sigma_psf=sigma_psf,
    )
    if sigma_psf <= 0:
        expected = FLUX
    else:
        expected = 2.0 * FLUX * (sigma**2 - sigma_psf**2) / sigma**2
    np.testing.assert_allclose(cat["flux_fpfs1"][0], expected, rtol=1e-10)


@pytest.mark.parametrize("sigma_w2", [0.0, 0.2**2])
@pytest.mark.parametrize(
    "shape", [(0.5, 0.5, 0.0), (0.7, 0.4, 0.0), (0.45, 0.6, 0.3)]
)
def test_reproduces_removed_gaussian_aperture_flux(sigma_w2, shape):
    """flux_gauss0 (window sigma_psf) and flux_gauss2 (window
    sqrt(sigma_psf^2 + 0.2^2)) of the removed GaussFit::measure_flux on
    the image reconvolved with sigma_psf = 0.38 are the fpfs1 flux of
    the kernel sqrt(sigma_psf^2 + sigma_meas^2) with that sigma_psf."""
    sigma_psf = 0.38
    sigma_meas2 = sigma_psf**2 + sigma_w2
    sigma = np.sqrt(sigma_psf**2 + sigma_meas2)
    # the FPFS stamp is float32: round the image once so both formulas
    # see the same pixels (the cast alone is a 1e-8 difference)
    img = _gaussian_image(*shape).astype(np.float32).astype(np.float64)
    old = _old_gaussian_flux(_reconvolve(img, sigma_psf), sigma_meas2)
    cat = _process(
        img, _delta(), _det_centre(),
        sigma_shapelets1=sigma, sigma_psf=sigma_psf, kmax=None,
    )
    np.testing.assert_allclose(cat["flux_fpfs1"][0], old, rtol=1e-8)


def test_adaptive_flux_response_carries_the_window_growth():
    """With a fixed sigma_psf only the window grows with the source's
    kernel size: flux = m00 4 pi (sigma^2 - sigma_psf^2) and
    dflux/dg = dm00/dg factor + m00 8 pi sigma dsigma/dg."""
    gal, psf = cosmos_scene()
    rng = np.random.RandomState(3)
    noise = rng.normal(scale=0.37, size=gal.shape)
    det0 = cosmos_centres()
    fit = anacal.fpfs.ADAPTIVE_FIT_COLUMNS
    det = np.zeros(
        len(det0), dtype=[("y", "f8"), ("x", "f8")] + [(c, "f8") for c in fit]
    )
    det["y"], det["x"] = det0["y"], det0["x"]
    det["mxx"] = 0.15 + 0.2 * rng.rand(len(det))
    det["myy"] = 0.15 + 0.2 * rng.rand(len(det))
    for c in fit[2:]:
        det[c] = rng.normal(0.0, 0.3, len(det))
    sigma_psf = 0.3
    cat = anacal.fpfs.process_image(
        fpfs_config=anacal.fpfs.FpfsConfig(
            sigma_shapelets1=0.52, adaptive_sigma=True, sigma_psf=sigma_psf
        ),
        mag_zero=30.0,
        gal_array=gal + rng.normal(scale=0.37, size=gal.shape),
        psf_array=psf,
        pixel_scale=SCALE,
        noise_variance=0.37**2,
        noise_array=noise,
        detection=det,
    )
    k = anacal.fpfs.ADAPTIVE_KERNEL
    sig = cat[f"{k}_sigma"]
    assert np.all(sig > sigma_psf)
    factor = 4.0 * np.pi * (sig**2 - sigma_psf**2)
    dfactor = 8.0 * np.pi * sig
    m00, dm1 = cat[f"{k}_m00"], cat[f"{k}_dm00_dg1"]
    np.testing.assert_allclose(cat[f"flux_{k}"], m00 * factor, rtol=1e-12)
    np.testing.assert_allclose(
        cat[f"dflux_{k}_dg1"],
        dm1 * factor + m00 * dfactor * cat[f"{k}_dsigma_dg1"],
        rtol=1e-12,
    )


def test_sigma_psf_errors():
    with pytest.raises(ValueError, match="sigma_psf"):
        anacal.fpfs.m00_to_flux(1.0, 0.5, 0.5)
    img = _gaussian_image(0.5, 0.5)
    with pytest.raises(ValueError, match="sigma_psf"):
        _process(
            img, _delta(), _det_centre(), sigma_shapelets1=0.5, sigma_psf=0.5
        )
    # the second kernel is the smaller one here
    with pytest.raises(ValueError, match="sigma_psf"):
        _process(
            img, _delta(), _det_centre(),
            sigma_shapelets1=0.7, sigma_shapelets2=0.45, sigma_psf=0.5,
        )
    with pytest.raises(ValueError, match="sigma_psf"):
        _process(
            img, _delta(), _det_centre(),
            sigma_shapelets1=0.7, adaptive_sigma=True,
            adaptive_sigma_floor=0.45, sigma_psf=0.5,
        )


# ---------------------------------------------------------------------
# The extra flux-only kernels (flux_gauss1/2) measured in the fpfs1 pass
# ---------------------------------------------------------------------
NOISE_FAMILY = ("flux_err", "s2n", "ds2n_dg1", "ds2n_dg2")
FAMILY = ("flux", "dflux_dg1", "dflux_dg2") + NOISE_FAMILY


def _cosmos_run(cfg, mode="rotate", with_noise=True, seed=1):
    gal, psf = cosmos_scene()
    rng = np.random.RandomState(seed)
    img = gal + rng.normal(scale=0.37, size=gal.shape)
    noise = rng.normal(scale=0.37, size=gal.shape)
    noise2 = rng.normal(scale=0.37, size=gal.shape)
    return anacal.fpfs.process_image(
        fpfs_config=cfg,
        mag_zero=30.0,
        gal_array=img,
        psf_array=psf,
        pixel_scale=SCALE,
        noise_variance=0.37**2,
        noise_array=noise if with_noise else None,
        noise2_array=noise2 if mode == "spectral" else None,
        renoise_mode=mode,
        detection=cosmos_centres(),
    )


def _fam_col(kernel, name):
    return anacal.fpfs._kernel_rename_map([name], kernel)[name]


@pytest.mark.parametrize(
    "mode, with_noise",
    [("rotate", True), ("rotate", False), ("spectral", True)],
)
def test_flux_gauss_kernels_match_fixed_kernels(mode, with_noise):
    """flux_gauss{i} is the flux family of a fixed kernel of size
    flux_gauss_sigmas[i] with sigma_psf = flux_gauss_sigma_psf: the
    same transforms, deconvolution, noise treatment and conversion, so
    flux and its responses are bit-identical; the noise family differs
    only by the summation order of its std_m00 (m00_std_grid)."""
    cfg = anacal.fpfs.FpfsConfig(sigma_shapelets1=0.53)
    assert cfg.flux_gauss_sigmas == (0.52, 0.65)
    assert cfg.flux_gauss_sigma_psf == 0.4
    cat = _cosmos_run(cfg, mode, with_noise)
    for i, sg in enumerate(cfg.flux_gauss_sigmas):
        ref = _cosmos_run(
            anacal.fpfs.FpfsConfig(
                sigma_shapelets1=sg, sigma_psf=0.4, flux_gauss_sigmas=()
            ),
            mode, with_noise,
        )
        for name in FAMILY:
            a = cat[_fam_col(f"gauss{i + 1}", name)]
            b = ref[_fam_col("fpfs1", name)]
            if name in NOISE_FAMILY:
                np.testing.assert_allclose(a, b, rtol=1e-12, err_msg=name)
            else:
                np.testing.assert_array_equal(a, b, err_msg=name)
    # the host kernel's own columns are untouched by the extras
    off = _cosmos_run(
        anacal.fpfs.FpfsConfig(sigma_shapelets1=0.53, flux_gauss_sigmas=()),
        mode, with_noise,
    )
    assert len(cat.dtype.names) == len(off.dtype.names) + 14
    assert cat.dtype.names[: len(off.dtype.names)] == off.dtype.names
    for name in off.dtype.names:
        np.testing.assert_array_equal(cat[name], off[name], err_msg=name)
    assert cat.dtype.names[len(off.dtype.names):] == (
        "flux_gauss1", "dflux_gauss1_dg1", "dflux_gauss1_dg2",
        "flux_gauss1_err", "s2n_gauss1", "ds2n_gauss1_dg1", "ds2n_gauss1_dg2",
        "flux_gauss2", "dflux_gauss2_dg1", "dflux_gauss2_dg2",
        "flux_gauss2_err", "s2n_gauss2", "ds2n_gauss2_dg1", "ds2n_gauss2_dg2",
    )


def test_flux_gauss_with_other_kernels_and_prefix():
    """The extras ride only in the fpfs1 pass: fpfs2 and the adaptive
    kernels carry none, the band prefix applies to them too, and an
    adaptive run leaves them unchanged."""
    cfg = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=0.53, sigma_shapelets2=0.6, flux_gauss_sigmas=(0.7,)
    )
    gal, psf = cosmos_scene()
    cat = anacal.fpfs.process_image(
        fpfs_config=cfg, mag_zero=30.0, gal_array=gal, psf_array=psf,
        pixel_scale=SCALE, noise_variance=1e-4, detection=cosmos_centres(),
        base_column_name="lsst_i_",
    )
    names = cat.dtype.names
    assert "lsst_i_flux_gauss1" in names
    assert "lsst_i_s2n_fpfs2" in names
    assert not any("gauss2" in n for n in names)
    assert sum("gauss1" in n for n in names) == 7


def test_flux_gauss_errors():
    gal, psf = cosmos_scene()
    kw = dict(
        mag_zero=30.0, gal_array=gal, psf_array=psf, pixel_scale=SCALE,
        noise_variance=1e-4, detection=cosmos_centres(),
    )
    with pytest.raises(ValueError, match="flux_gauss_sigma_psf"):
        anacal.fpfs.process_image(
            fpfs_config=anacal.fpfs.FpfsConfig(
                sigma_shapelets1=0.53, flux_gauss_sigmas=(0.4,),
                flux_gauss_sigma_psf=0.4,
            ),
            **kw,
        )
    with pytest.raises(ValueError, match="at most four"):
        anacal.fpfs.process_image(
            fpfs_config=anacal.fpfs.FpfsConfig(
                sigma_shapelets1=0.53, flux_gauss_sigmas=(0.5,) * 5,
            ),
            **kw,
        )
