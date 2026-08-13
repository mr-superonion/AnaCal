import anacal
import numpy as np
import pytest


def test_mask():
    ngrid = 128
    sigma_arcsec = 0.52
    scale = 0.2
    mask = np.zeros((ngrid, ngrid), dtype=np.uint8)
    mask[ngrid // 2 + 10, ngrid // 2 - 20] = 1
    b = anacal.mask.convolve_mask_gauss(
        mask,
        sigma=sigma_arcsec,
        scale=scale,
    )
    np.testing.assert_almost_equal(np.sum(b), 1, decimal=1)
    assert b[ngrid // 2 + 10, ngrid // 2 - 20] == np.max(b)

    # add_pixel_mask_column returns the updated catalog (the input list is a
    # copy at the pybind boundary, so in-place mutation would be lost).
    # It must reproduce sampling the convolve_mask_gauss image exactly,
    # while only READING the mask (the smoothed value is evaluated at each
    # source position; no smoothed copy of the whole image is built).
    on_mask = anacal.table.galNumber()
    on_mask.model.x1 = anacal.math.qnumber((ngrid // 2 - 20) * scale)
    on_mask.model.x2 = anacal.math.qnumber((ngrid // 2 + 10) * scale)
    off_mask = anacal.table.galNumber()
    off_mask.model.x1 = anacal.math.qnumber(5 * scale)
    off_mask.model.x2 = anacal.math.qnumber(5 * scale)
    mask_before = mask.copy()
    out = anacal.mask.add_pixel_mask_column(
        [on_mask, off_mask], mask, sigma_arcsec, scale
    )
    assert out[0].mask_value == int(
        b[ngrid // 2 + 10, ngrid // 2 - 20] * 1000
    )
    assert out[0].mask_value > 0
    assert out[1].mask_value == 0
    np.testing.assert_array_equal(mask, mask_before)
    star_array = np.array(
        [
            (10.0, 3.0, 20.0),
            (22.0, 55.0, 20.0),
            (65.0, 80.0, 20.0),
            (120.0, 1.0, 20.0),
        ],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("r", "f4"),
        ],
    )
    anacal.mask.add_bright_star_mask(mask, star_array)
    assert mask[4, 11] == 1
    assert mask[55, 41] == 1
    assert mask[99, 65] == 1

    mask = np.zeros((ngrid, ngrid), dtype=np.uint8)
    mask[ngrid // 2 + 10, ngrid // 2 - 20] = 1
    # float32, as the surveys store their science planes -- this one is edited
    # in place, so AnaCal refuses any other dtype rather than quietly writing
    # into a converted copy and leaving the caller's pixels alone.
    data = np.ones((ngrid, ngrid), dtype=np.float32) * 10.0
    anacal.mask.mask_galaxy_image(data, mask, star_array)
    assert data[4, 11] == 0
    assert data[55, 41] == 0
    assert data[99, 65] == 0

    with pytest.raises(ValueError, match="must already be float32"):
        anacal.mask.mask_galaxy_image(
            np.ones((ngrid, ngrid), dtype=np.float64), mask, star_array
        )

    # Both arrays are edited in place, so a wrong mask dtype must RAISE.
    # It used to be accepted silently: pybind converted it to a copy,
    # flagged the copy, and handed the caller back an untouched mask
    # with no error.
    with pytest.raises(ValueError, match="must already be uint8"):
        anacal.mask.add_bright_star_mask(
            np.ones((ngrid, ngrid)), star_array
        )
    with pytest.raises(ValueError, match="must already be uint8"):
        anacal.mask.mask_galaxy_image(
            np.ones((ngrid, ngrid), dtype=np.float32),
            np.ones((ngrid, ngrid)),
            star_array,
        )

    # A uint8 mask really is written through to the caller's buffer.
    mask = np.zeros((ngrid, ngrid), dtype=np.uint8)
    anacal.mask.add_bright_star_mask(mask, star_array)
    assert mask[4, 11] == 1

    mask = np.ones((ngrid, ngrid), dtype=np.uint8)
    b = anacal.mask.convolve_mask_gauss(
        mask,
        sigma=sigma_arcsec,
        scale=scale,
    )
    np.testing.assert_array_less(b, np.ones((ngrid, ngrid)))
    np.testing.assert_array_less(-b, -0.3 * np.ones((ngrid, ngrid)))

    return


def test_add_bright_star_mask_min_radius():
    ngrid = 32
    mask = np.zeros((ngrid, ngrid), dtype=np.uint8)
    stars = np.array(
        [(4.49, 5.49, 0.2)],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("r", "f4"),
        ],
    )

    anacal.mask.add_bright_star_mask(mask, stars)

    assert mask[5, 4] == 1
    assert mask.sum() == 1


def test_mask_bits():
    """Bit 0 (masked) and bit 1 (discontinuity) are independent channels."""
    scale = 1.0
    sigma = 1.5
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:15, 10:15] = 1
    mask[10:15, 25:30] = 2
    mask[30:32, 30:32] = 3

    def src(x, y):
        g = anacal.table.galNumber()
        g.model.x1 = anacal.math.qnumber(x * scale)
        g.model.x2 = anacal.math.qnumber(y * scale)
        return g

    out = anacal.mask.add_pixel_mask_column(
        [src(12, 12), src(27, 12), src(5, 5), src(30.5, 30.5)],
        mask, sigma, scale,
    )
    assert out[0].mask_value > 0 and out[0].discontinuity_mask_value == 0
    assert out[1].mask_value == 0 and out[1].discontinuity_mask_value > 0
    assert out[2].mask_value == 0 and out[2].discontinuity_mask_value == 0
    # symmetric blobs at symmetric offsets: identical smoothed values
    assert out[0].mask_value == out[1].discontinuity_mask_value
    assert out[3].mask_value == out[3].discontinuity_mask_value > 0

    # the galaxy image loses bit-0 pixels only
    gal = np.ones((40, 40), dtype=np.float32)
    m3 = mask.copy()
    anacal.mask.mask_galaxy_image(gal, m3, None)
    assert gal[12, 12] == 0.0 and gal[30, 30] == 0.0
    assert gal[12, 27] == 1.0

    # the in-place mutators refuse non-uint8 masks instead of silently
    # writing into a converted copy
    with pytest.raises(ValueError):
        anacal.mask.mask_galaxy_image(gal, mask.astype(np.int16), None)
