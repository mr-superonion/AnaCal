"""Detection on several bands at once.

The bands are combined AFTER each band's PSF has been removed, so the
combination is an average of images that already share the same effective
Gaussian PSF.  These tests pin down the two things that can silently go wrong:
the weighting of the bands, and the noise level of the result.
"""

import anacal
import galsim
import numpy as np
import pytest

SCALE = 0.2
SIGMA_ARCSEC = 0.4
STAMP = 32
NGAL = 6
VARIANCE = 0.10

KWARGS = {
    "omega_f": 0.8,
    "omega_v": 0.04,
}


def make_psf(fwhm):
    obj = galsim.Moffat(beta=2.5, fwhm=fwhm).shear(g1=0.02, g2=-0.02)
    return obj, (
        obj.shift(0.5 * SCALE, 0.5 * SCALE)
        .drawImage(nx=STAMP, ny=STAMP, scale=SCALE)
        .array
    )


def make_image(psf_obj, mag, seed=None):
    """One tiled field of identical galaxies, optionally with noise."""
    flux = 10 ** ((30.0 - mag) / 2.5)
    gal = galsim.Exponential(half_light_radius=0.30).shear(g1=0.03)
    gal = galsim.Convolve(psf_obj, gal.withFlux(flux))

    stamp = galsim.ImageF(ncol=STAMP, nrow=STAMP, scale=SCALE)
    gal.shift(
        galsim.PositionD(0.5 * SCALE, 0.5 * SCALE)
    ).drawImage(image=stamp)
    img = np.tile(stamp.array, (NGAL, NGAL)).astype(np.float32)
    if seed is not None:
        rng = np.random.RandomState(seed)
        img = img + rng.normal(
            scale=np.sqrt(VARIANCE), size=img.shape
        ).astype(np.float32)
    return img


def make_task():
    prior = anacal.ngmix.modelPrior()
    prior.set_sigma_x(anacal.math.qnumber(0.5))
    return anacal.task.Task(
        scale=SCALE,
        sigma_arcsec=SIGMA_ARCSEC,
        snr_peak_min=10,
        stamp_size=STAMP,
        image_bound=0,
        num_epochs=10,
        prior=prior,
        force_size=True,
        **KWARGS,
    )


def make_blocks(shape):
    return anacal.geometry.get_block_list(
        shape[0], shape[1], 500, 500, 64, SCALE,
    )


def run(img, psf, variance):
    task = make_task()
    return task.process_image(
        img, psf, variance=variance, block_list=make_blocks(img.shape[-2:]),
    )


def assert_catalogs_identical(a, b):
    assert len(a) == len(b)
    assert a.dtype.names == b.dtype.names
    for name in a.dtype.names:
        np.testing.assert_array_equal(
            a[name], b[name], err_msg=f"column '{name}' differs",
        )


def assert_catalogs_match(a, b, tol=1e-9):
    """Same catalog to round-off.

    Columns are compared against the scale of the column as a whole, not
    value by value: several derivative columns are zero for a round shear,
    and a relative test on those would be measuring nothing but noise.
    """
    assert len(a) == len(b)
    assert a.dtype.names == b.dtype.names
    for name in a.dtype.names:
        va = np.asarray(a[name], dtype=np.float64)
        vb = np.asarray(b[name], dtype=np.float64)
        scale = np.sqrt(np.mean(va ** 2))
        assert np.max(np.abs(vb - va)) <= tol * max(scale, 1.0), (
            f"column '{name}' differs by more than round-off"
        )


def test_one_band_stack_matches_plain_image():
    """A stack of one band must be the plain 2-D call, bit for bit."""
    psf_obj, psf = make_psf(0.7)
    img = make_image(psf_obj, 23.5)

    plain = run(img, psf, VARIANCE)
    stacked = run(img[None, :, :], psf[None, :, :], [VARIANCE])

    assert len(plain) == NGAL * NGAL
    assert_catalogs_identical(plain, stacked)


def test_three_identical_bands_reproduce_one_band():
    """Three copies of one band, each with three times the variance.

    The weights come out at 1/3 each, so the average is the original image,
    and the combined variance is 3 * (1/3)^2 * 3V = V.  Anything beyond
    round-off means the weighting or the variance propagation is wrong.

    Round-off is all that separates them: adding q/3 three times does not
    land exactly back on q, so this is allclose rather than an exact match.
    """
    psf_obj, psf = make_psf(0.7)
    img = make_image(psf_obj, 23.5)

    single = run(img, psf, VARIANCE)
    tripled = run(
        np.repeat(img[None, :, :], 3, axis=0),
        np.repeat(psf[None, :, :], 3, axis=0),
        [3.0 * VARIANCE] * 3,
    )
    assert_catalogs_match(single, tripled, tol=1e-10)


def test_band_order_does_not_matter():
    psf_objs, psfs, imgs, variances = [], [], [], []
    for fwhm, mag, var in ((0.7, 23.5, 0.10), (0.9, 24.0, 0.25), (0.6, 24.5, 0.40)):
        obj, psf = make_psf(fwhm)
        psf_objs.append(obj)
        psfs.append(psf)
        imgs.append(make_image(obj, mag))
        variances.append(var)

    order = [2, 0, 1]
    forward = run(np.array(imgs), np.array(psfs), variances)
    reverse = run(
        np.array([imgs[i] for i in order]),
        np.array([psfs[i] for i in order]),
        [variances[i] for i in order],
    )

    assert len(forward) == len(reverse)
    for name in forward.dtype.names:
        np.testing.assert_allclose(
            forward[name], reverse[name], rtol=1e-8, atol=1e-10,
            err_msg=f"column '{name}' depends on band order",
        )


def test_extra_bands_deepen_the_detection_image():
    """Adding bands must lower the noise of the detection image."""
    psf_obj, psf = make_psf(0.7)
    variances = [VARIANCE, 2.0 * VARIANCE, 3.0 * VARIANCE]
    psfs = np.repeat(psf[None, :, :], 3, axis=0)

    sigma_det = SIGMA_ARCSEC * np.sqrt(2.0)
    std = []
    for nband in (1, 2, 3):
        w = anacal.detector.band_weights(
            SCALE, sigma_det, psfs[:nband], variances[:nband],
        )
        np.testing.assert_allclose(np.sum(w), 1.0, rtol=1e-12)
        std.append(
            anacal.detector.coadd_smoothed_variance(
                SCALE, sigma_det, psfs[:nband], variances[:nband], w,
            )
        )
    assert std[0] > std[1] > std[2]

    # Same sources, no noise: the flux is unchanged but its error shrinks as
    # bands are added.
    clean = np.repeat(make_image(psf_obj, 23.5)[None, :, :], 3, axis=0)
    errs, fluxes = [], []
    for nband in (1, 2, 3):
        cat = run(clean[:nband], psfs[:nband], variances[:nband])
        assert len(cat) == NGAL * NGAL
        errs.append(float(cat["flux_gauss0_err"][0]))
        fluxes.append(float(np.mean(cat["flux_gauss0"])))
    assert errs[0] > errs[1] > errs[2]
    np.testing.assert_allclose(fluxes[1:], fluxes[0], rtol=1e-10)

    # And faint sources that a single band misses are picked up by three.
    imgs = np.array([make_image(psf_obj, 26.0, seed=100 + i) for i in range(3)])
    one = run(imgs[:1], psfs[:1], variances[:1])
    three = run(imgs, psfs, variances)
    assert len(one) < len(three) <= NGAL * NGAL
    for name in three.dtype.names:
        assert np.all(np.isfinite(three[name])), (
            f"column '{name}' has non-finite values"
        )


def test_shape_and_variance_errors():
    psf_obj, psf = make_psf(0.7)
    img = make_image(psf_obj, 23.5)
    stack2 = np.repeat(img[None, :, :], 2, axis=0)
    psf2 = np.repeat(psf[None, :, :], 2, axis=0)

    with pytest.raises(RuntimeError, match="band"):
        # two image bands, three PSF bands
        run(stack2, np.repeat(psf[None, :, :], 3, axis=0), [VARIANCE] * 2)

    with pytest.raises(RuntimeError, match="variance value"):
        run(stack2, psf2, [VARIANCE])

    with pytest.raises(RuntimeError, match="positive and finite"):
        run(stack2, psf2, [VARIANCE, 0.0])

    with pytest.raises(RuntimeError, match="band"):
        # a 3-D image with a 2-D PSF
        run(stack2, psf, [VARIANCE] * 2)

    with pytest.raises(RuntimeError, match="dimensions"):
        run(img[None, None, :, :], psf2, [VARIANCE] * 2)
