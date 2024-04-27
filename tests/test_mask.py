import anacal
import numpy as np


def test_mask():
    ngrid = 128
    sigma_arcsec = 0.52
    scale = 0.2
    bound = 25
    mask = np.zeros((ngrid, ngrid), dtype=np.int16)
    mask[ngrid // 2 + 10, ngrid // 2 - 20] = 1
    b = anacal.mask.smooth_mask_image(
        mask, sigma=sigma_arcsec, scale=scale, bound=bound,
    )
    np.testing.assert_almost_equal(np.sum(b), 1, decimal=1)
    assert b[ngrid // 2 + 10, ngrid // 2 - 20] == np.max(b)

    src = np.array([
        [ngrid // 2 + 10, ngrid // 2 - 20, 0, 0]
    ]).astype(np.int32)
    anacal.mask.add_pixel_mask_column(src, mask, sigma_arcsec, scale, bound)
    assert src[0, 3] > 0
    x_array = np.array([10, 22, 65, 120])
    y_array = np.array([3, 55, 80, 1])
    r_array = np.array([20, 20, 20, 20])
    anacal.mask.mask_bright_stars(mask, x_array, y_array, r_array)
    # import fitsio
    # fitsio.write("mask.fits", mask)
    assert mask[4, 11] == 16
    assert mask[55, 41] == 16
    assert mask[99, 65] == 16
    return

