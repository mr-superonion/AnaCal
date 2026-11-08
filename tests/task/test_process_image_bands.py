"""Task.process_image_bands: the per-band catalogs of a band-coadd
detection, measured from the band images the coadd was built from, must
be the forced single-band measurement of that catalog -- bit for bit, in
every column -- in every mode (plain, rotated noise, spectral renoising,
masked, external detection), while the coadd catalog is unchanged."""

import anacal
import numpy as np
import pytest

from .test_multiband import (
    NGAL,
    SCALE,
    VARIANCE,
    assert_catalogs_identical,
    make_cells,
    make_image,
    make_psf,
    make_task,
)

BAND_SPECS = ((0.7, 23.5, 0.10), (0.9, 24.0, 0.25), (0.6, 24.5, 0.40))


def _bands(seed=None):
    imgs, psfs, variances = [], [], []
    for i, (fwhm, mag, var) in enumerate(BAND_SPECS):
        obj, psf = make_psf(fwhm)
        psfs.append(psf)
        imgs.append(
            make_image(obj, mag, seed=None if seed is None else seed + i)
        )
        variances.append(var)
    return np.array(imgs), np.array(psfs), variances


def _pure_noise(shape, variances, seed):
    rng = np.random.RandomState(seed)
    return np.array([
        rng.normal(scale=np.sqrt(v), size=shape).astype(np.float32)
        for v in variances
    ])


def assert_band_is_forced(band_cat, forced):
    """The band catalog IS the forced single-band measurement, bit for
    bit, in every column."""
    assert_catalogs_identical(band_cat, forced)


def test_bands_are_the_forced_single_band_measurements():
    imgs, psfs, variances = _bands()
    task = make_task()
    cells = make_cells(imgs.shape[-2:])
    cat, bands = task.process_image_bands(
        imgs, psfs, variance=variances, cell_list=cells,
    )
    # the coadd catalog is process_image's, untouched
    ref = task.process_image(imgs, psfs, variance=variances, cell_list=cells)
    assert len(cat) == NGAL * NGAL
    assert_catalogs_identical(cat, ref)
    assert len(bands) == len(BAND_SPECS)
    for b in range(len(BAND_SPECS)):
        forced = task.process_image(
            imgs[b], psfs[b], variance=variances[b], cell_list=cells,
            detection=cat,
        )
        assert_band_is_forced(bands[b], forced)
    # the bands really differ (different depth and PSF), so a mix-up or a
    # copy of the coadd could not pass the comparison above
    b0, b1 = bands[0]["flux_gauss0"], bands[1]["flux_gauss0"]
    assert not np.array_equal(b0, b1)
    assert not np.array_equal(b0, cat["flux_gauss0"])
    assert np.all(bands[0]["flux_gauss0_err"] != bands[1]["flux_gauss0_err"])


def test_plain_image_gives_one_band():
    obj, psf = make_psf(0.7)
    img = make_image(obj, 23.5)
    task = make_task()
    cells = make_cells(img.shape)
    cat, bands = task.process_image_bands(
        img, psf, variance=VARIANCE, cell_list=cells,
    )
    assert len(bands) == 1
    assert_catalogs_identical(
        cat, task.process_image(img, psf, variance=VARIANCE, cell_list=cells),
    )
    forced = task.process_image(
        img, psf, variance=VARIANCE, cell_list=cells, detection=cat,
    )
    assert_band_is_forced(bands[0], forced)


def test_bands_with_rotated_noise_mask_and_cells():
    """Rotated-noise renoising, a mask with the n_mask_base cut and more
    than one cell: the per-band rows follow the coadd rows cell by cell."""
    imgs, psfs, variances = _bands(seed=3)
    noise = _pure_noise(imgs.shape[-2:], variances, seed=30)
    n = imgs.shape[-1]
    cells = anacal.geometry.get_cell_list(
        img_nx=n, img_ny=n, cell_nx=128, cell_ny=128, cell_overlap=32,
        scale=SCALE,
    )
    assert len(cells) > 1
    mask = np.zeros(imgs.shape[-2:], dtype=np.uint8)
    mask[8:40, 8:40] = 1  # the first grid galaxy sits on masked pixels
    task = make_task()
    kw = dict(mask_array=mask, n_mask_base_max=0.1)
    cat, bands = task.process_image_bands(
        imgs, psfs, variance=variances, cell_list=cells, noise_array=noise,
        **kw,
    )
    ref = task.process_image(
        imgs, psfs, variance=variances, cell_list=cells, noise_array=noise,
        **kw,
    )
    assert_catalogs_identical(cat, ref)
    assert len(cat) > 0
    assert np.any(cat["n_mask_base"] > 0.1)  # the cut is exercised
    for b in range(len(BAND_SPECS)):
        forced = task.process_image(
            imgs[b], psfs[b], variance=variances[b], cell_list=cells,
            detection=cat, noise_array=noise[b], **kw,
        )
        assert_band_is_forced(bands[b], forced)


@pytest.mark.parametrize("mode", ["rotate", "spectral"])
def test_bands_with_renoise_setup(mode):
    """A renoise Setup in either mode: the flux errors are summed over
    the per-band spectra in both, the kernels act on the measurement in
    the spectral mode only."""
    imgs, psfs, variances = _bands(seed=5)
    noise = _pure_noise(imgs.shape[-2:], variances, seed=50)
    noise2 = _pure_noise(imgs.shape[-2:], variances, seed=51)
    if mode == "rotate":
        noise2 = None
    task = make_task()
    cells = make_cells(imgs.shape[-2:])
    n = int(cells[0].nx)
    spectra = [
        anacal.renoise.make_noise_spectrum(n, SCALE, v, None) for v in variances
    ]

    def setup(spec):
        return anacal.renoise.Setup(
            c=0.1, spectra=spec, noise_rotated=True, mode=mode, delta=0.01,
        )

    cat, bands = task.process_image_bands(
        imgs, psfs, variance=variances, cell_list=cells, noise_array=noise,
        noise2_array=noise2, renoise=setup(spectra),
    )
    ref = task.process_image(
        imgs, psfs, variance=variances, cell_list=cells, noise_array=noise,
        noise2_array=noise2, renoise=setup(spectra),
    )
    assert_catalogs_identical(cat, ref)
    assert len(cat) > 0
    for b in range(len(BAND_SPECS)):
        forced = task.process_image(
            imgs[b], psfs[b], variance=variances[b], cell_list=cells,
            detection=cat, noise_array=noise[b],
            noise2_array=None if noise2 is None else noise2[b],
            renoise=setup([spectra[b]]),
        )
        assert_band_is_forced(bands[b], forced)
    # with a Setup the errors come from the spectra, not the white-noise
    # formula: a run without the Setup gives different errors
    plain = task.process_image(
        imgs[0], psfs[0], variance=variances[0], cell_list=cells,
        detection=cat, noise_array=noise[0],
    )
    assert not np.array_equal(
        plain["flux_gauss0_err"], bands[0]["flux_gauss0_err"]
    )


def test_external_detection_keeps_the_input_order():
    imgs, psfs, variances = _bands()
    task = make_task()
    cells = make_cells(imgs.shape[-2:])
    det = task.process_image(imgs, psfs, variance=variances, cell_list=cells)
    perm = np.random.RandomState(1).permutation(len(det))
    det = det[perm]
    cat, bands = task.process_image_bands(
        imgs, psfs, variance=variances, cell_list=cells, detection=det,
    )
    np.testing.assert_array_equal(cat["x1_det"], det["x1_det"])
    assert_catalogs_identical(
        cat,
        task.process_image(
            imgs, psfs, variance=variances, cell_list=cells, detection=det,
        ),
    )
    for b in range(len(BAND_SPECS)):
        np.testing.assert_array_equal(bands[b]["x1_det"], det["x1_det"])
        forced = task.process_image(
            imgs[b], psfs[b], variance=variances[b], cell_list=cells,
            detection=cat,
        )
        assert_band_is_forced(bands[b], forced)
