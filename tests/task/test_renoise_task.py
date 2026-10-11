"""The renoise Setup of anacal.task.Task.process_image.

Option 1 ("rotate") with a white noise spectrum must reproduce the plain
rotated-noise path (renoise=None, the code before the noise spectrum
existed, which doubles the noise variance), and inconsistent setups must
be rejected.  That the Task's measurement under either mode equals the
forced FPFS measurement is tested in
tests/fpfs/test_task_force_consistency.py.
"""

import anacal
import numpy as np
import pytest

from ..fixtures import cosmos_scene, symmetrized_spectrum

SCALE = 0.2
KLIM = anacal.task.KLIM_PIX / SCALE
NOISE_STD = 0.37


def make_scene(seed=11):
    gal, psf = cosmos_scene()
    rng = np.random.default_rng(seed)
    img = (gal + rng.normal(0.0, NOISE_STD, gal.shape)).astype(np.float32)
    noise = rng.normal(0.0, NOISE_STD, gal.shape).astype(np.float32)
    return img, psf, noise, NOISE_STD**2


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


def test_rotate_setup_reproduces_plain_path():
    """A rotate-mode Setup with the white spectrum (noise_corr=None) gives
    the catalog of the plain path: same detections and every column equal
    to roundoff (the variances come from a spectrum sum instead of the
    factor 2, so not bit for bit)."""
    img, psf, noise, var = make_scene()
    cells = cells_of(img)
    task = make_task(psf)
    cat0 = task.process_image(
        img, psf, variance=var, cell_list=cells, noise_array=noise
    )
    raw = anacal.renoise.make_noise_spectrum(cells[0].nx, SCALE, var, None)
    setup = anacal.renoise.Setup(
        spectra=[raw], noise_rotated=True, mode="rotate"
    )
    cat1 = task.process_image(
        img,
        psf,
        variance=var,
        cell_list=cells,
        noise_array=noise,
        renoise=setup,
    )
    assert len(cat0) >= 90 and len(cat1) == len(cat0)
    for col in cat0.dtype.names:
        if cat0[col].dtype.kind == "f":
            np.testing.assert_allclose(
                cat1[col], cat0[col], rtol=1e-10, atol=1e-12, err_msg=col
            )
        else:
            assert np.array_equal(cat1[col], cat0[col]), col


def test_renoise_setup_errors():
    img, psf, noise, var = make_scene()
    cells = cells_of(img)
    n = cells[0].nx
    task = make_task(psf)
    sym = symmetrized_spectrum(n, SCALE, var, psf, KLIM)
    raw = anacal.renoise.make_noise_spectrum(n, SCALE, var, None)
    # any Setup needs the pure-noise image
    for setup in (
        anacal.renoise.Setup(spectra=[sym]),
        anacal.renoise.Setup(spectra=[raw], mode="rotate"),
    ):
        with pytest.raises(RuntimeError):
            task.process_image(
                img, psf, variance=var, cell_list=cells, renoise=setup
            )
    # the spectrum must match the mode
    for setup in (
        anacal.renoise.Setup(spectra=[raw], mode="spectral"),
        anacal.renoise.Setup(spectra=[sym], mode="rotate"),
    ):
        with pytest.raises(RuntimeError):
            task.process_image(
                img,
                psf,
                variance=var,
                cell_list=cells,
                noise_array=noise,
                renoise=setup,
            )
    # wrong grid
    bad = anacal.renoise.Setup(
        spectra=[symmetrized_spectrum(n + 2, SCALE, var, psf, KLIM)]
    )
    with pytest.raises(RuntimeError):
        task.process_image(
            img,
            psf,
            variance=var,
            cell_list=cells,
            noise_array=noise,
            renoise=bad,
        )
    # wrong number of bands
    two = anacal.renoise.Setup(spectra=[sym] * 2)
    with pytest.raises(RuntimeError):
        task.process_image(
            img,
            psf,
            variance=var,
            cell_list=cells,
            noise_array=noise,
            renoise=two,
        )
