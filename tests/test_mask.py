import anacal
import numpy as np


def test_mask():
    ngrid = 128
    sigma_arcsec = 0.52
    scale = 0.2
    mask = np.zeros((ngrid, ngrid), dtype=np.int16)
    mask[ngrid // 2 + 10, ngrid // 2 - 20] = 1
    b = anacal.mask.smooth_mask_image(
        mask,
        sigma=sigma_arcsec,
        scale=scale,
    )
    np.testing.assert_almost_equal(np.sum(b), 1, decimal=1)
    assert b[ngrid // 2 + 10, ngrid // 2 - 20] == np.max(b)

    src = np.array(
        [(ngrid // 2 + 10, ngrid // 2 - 20, True, 0)],
        dtype=[
            ("y", np.int32),
            ("x", np.int32),
            ("is_peak", np.int32),
            ("mask_value", np.int32),
        ],
    )
    src = anacal.mask.add_pixel_mask_column(src, mask, sigma_arcsec, scale)
    assert src["mask_value"][0] == 23
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
    data = np.ones((ngrid, ngrid)) * 10.0
    anacal.mask.mask_galaxy_image(data, mask, True, star_array)
    assert data[4, 11] == 0
    assert data[55, 41] == 0
    assert data[99, 65] == 0

    mask = np.ones((ngrid, ngrid))
    anacal.mask.extend_mask_image(mask)
    anacal.mask.add_bright_star_mask(mask, star_array)
    np.testing.assert_almost_equal(mask, np.ones((ngrid, ngrid)))
    mask = np.ones((ngrid, ngrid))
    b = anacal.mask.smooth_mask_image(
        mask,
        sigma=sigma_arcsec,
        scale=scale,
    )
    np.testing.assert_array_less(b, np.ones((ngrid, ngrid)))
    np.testing.assert_array_less(-b, -0.3 * np.ones((ngrid, ngrid)))

    return
