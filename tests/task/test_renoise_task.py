"""Option-2 (spectral) renoising through anacal.task.Task.process_image.

The default path (renoise=None) is covered by the other task tests; here
the spectral path must run end to end on a noisy scene whose noise carries
the spin-2 filler of the patch PSF, give finite numbers, detect the same
sources as the rotated-noise path to within the Monte Carlo scatter, follow
the variance bookkeeping, and reject inconsistent setups.
"""

import anacal
import numpy as np
import pytest

from ..fixtures import load, make_filler, symmetrized_spectrum

SCALE = 0.2
KLIM = 3.1 / SCALE
C, DELTA = 0.1, 0.05


def make_scene(seed=11):
    fix = load("fpfs_process_image")
    psf = np.asarray(fix["psf"], dtype=np.float64)
    gal = np.asarray(fix["gal"], dtype=np.float64)
    noise_std = 0.37
    var = noise_std**2
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, noise_std, gal.shape)
    eps2 = rng.normal(0.0, noise_std, gal.shape)
    noise = rng.normal(0.0, noise_std, gal.shape).astype(np.float32)
    img = (gal + eps).astype(np.float32)
    # the image for the spectral path: its noise carries the filler
    img_s = (
        gal + eps + make_filler(eps2, SCALE, var, psf, KLIM, DELTA)
    ).astype(np.float32)
    return img, img_s, psf, noise, var


def make_task(psf):
    return anacal.task.Task(
        scale=SCALE,
        sigma_arcsec=0.52,
        snr_peak_min=8.0,
        omega_f=0.218,
        omega_v=0.011,
        stamp_size=psf.shape[0],
        num_epochs=10,
    )


def cells_of(img):
    return anacal.geometry.get_cell_list(
        img_nx=img.shape[1],
        img_ny=img.shape[0],
        cell_nx=250,
        cell_ny=250,
        cell_overlap=64,
        scale=SCALE,
    )


def test_task_spectral_renoising():
    img, img_s, psf, noise, var = make_scene()
    cells = cells_of(img)
    n = cells[0].nx
    assert all(c.nx == n and c.ny == n for c in cells)
    task = make_task(psf)
    cat1 = task.process_image(
        img, psf, variance=var, cell_list=cells, noise_array=noise
    )
    sym = symmetrized_spectrum(n, SCALE, var, psf, KLIM, DELTA)
    setup = anacal.renoise.Setup(c=C, spectra=[sym], noise_rotated=False)
    cat2 = task.process_image(
        img_s,
        psf,
        variance=var,
        cell_list=cells,
        noise_array=noise,
        renoise=setup,
    )
    for cat in (cat1, cat2):
        assert len(cat) > 50
        for col in ("e1", "de1_dg1", "flux_gauss0", "flux_gauss0_err"):
            assert np.all(np.isfinite(cat[col]))
    assert abs(len(cat2) - len(cat1)) < 0.2 * len(cat1), (len(cat1), len(cat2))
    # flux error: Option 1 doubles the variance, Option 2 scales it by the
    # variance factor of the symmetrized plus added noise (well below 2
    # for this nearly round PSF)
    r = np.median(cat2["flux_gauss0_err"]) / np.median(cat1["flux_gauss0_err"])
    rk = anacal.renoise.make_renoise_kernels(sym, psf, KLIM, C, 0.0, False)
    pred = np.sqrt(anacal.renoise.variance_factor(rk, 0.52) / 2.0)
    np.testing.assert_allclose(r, pred, rtol=0.02)
    assert pred < 0.85


def test_task_spectral_renoising_errors():
    img, img_s, psf, noise, var = make_scene()
    cells = cells_of(img)
    n = cells[0].nx
    task = make_task(psf)
    setup = anacal.renoise.Setup(
        spectra=[symmetrized_spectrum(n, SCALE, var, psf, KLIM)]
    )
    # the spectral mode needs the pure-noise image
    with pytest.raises(RuntimeError):
        task.process_image(
            img_s, psf, variance=var, cell_list=cells, renoise=setup
        )
    # wrong grid
    bad = anacal.renoise.Setup(
        spectra=[symmetrized_spectrum(n + 2, SCALE, var, psf, KLIM)]
    )
    with pytest.raises(RuntimeError):
        task.process_image(
            img_s,
            psf,
            variance=var,
            cell_list=cells,
            noise_array=noise,
            renoise=bad,
        )
    # wrong number of bands
    two = anacal.renoise.Setup(
        spectra=[symmetrized_spectrum(n, SCALE, var, psf, KLIM)] * 2
    )
    with pytest.raises(RuntimeError):
        task.process_image(
            img_s,
            psf,
            variance=var,
            cell_list=cells,
            noise_array=noise,
            renoise=two,
        )
