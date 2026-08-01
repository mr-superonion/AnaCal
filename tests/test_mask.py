import anacal
import numpy as np
import pytest


def test_mask():
    ngrid = 128
    sigma_arcsec = 0.52
    scale = 0.2
    mask = np.zeros((ngrid, ngrid), dtype=np.int16)
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
    on_mask = anacal.table.galNumber()
    on_mask.model.x1 = anacal.math.qnumber((ngrid // 2 - 20) * scale)
    on_mask.model.x2 = anacal.math.qnumber((ngrid // 2 + 10) * scale)
    off_mask = anacal.table.galNumber()
    off_mask.model.x1 = anacal.math.qnumber(5 * scale)
    off_mask.model.x2 = anacal.math.qnumber(5 * scale)
    out = anacal.mask.add_pixel_mask_column(
        [on_mask, off_mask], mask, sigma_arcsec, scale
    )
    assert out[0].mask_value == int(
        b[ngrid // 2 + 10, ngrid // 2 - 20] * 1000
    )
    assert out[0].mask_value > 0
    assert out[1].mask_value == 0
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

    mask = np.zeros((ngrid, ngrid), dtype=np.int16)
    mask[ngrid // 2 + 10, ngrid // 2 - 20] = 1
    # float32, as the surveys store their science planes -- this one is edited
    # in place, so AnaCal refuses any other dtype rather than quietly writing
    # into a converted copy and leaving the caller's pixels alone.
    data = np.ones((ngrid, ngrid), dtype=np.float32) * 10.0
    anacal.mask.mask_galaxy_image(data, mask, True, star_array)
    assert data[4, 11] == 0
    assert data[55, 41] == 0
    assert data[99, 65] == 0

    with pytest.raises(ValueError, match="must already be float32"):
        anacal.mask.mask_galaxy_image(
            np.ones((ngrid, ngrid), dtype=np.float64), mask, True, star_array
        )

    mask = np.ones((ngrid, ngrid))
    anacal.mask.extend_mask_image(mask)
    anacal.mask.add_bright_star_mask(mask, star_array)
    np.testing.assert_almost_equal(mask, np.ones((ngrid, ngrid)))
    mask = np.ones((ngrid, ngrid))
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
    mask = np.zeros((ngrid, ngrid), dtype=np.int16)
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
