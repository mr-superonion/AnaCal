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
    anacal.mask.add_pixel_mask_value(src, mask, sigma_arcsec, scale, bound)
    assert src[0, 3] > 0
    return

