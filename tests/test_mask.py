import anacal
import numpy as np


def test_mask():
    ngrid = 128
    a = np.zeros((ngrid, ngrid))
    a[ngrid // 2 + 10, ngrid // 2 - 20] = 1
    b = anacal.mask.smooth_mask_image(a, sigma=0.52, scale=0.2, bound=25)
    np.testing.assert_almost_equal(np.sum(b), 1, decimal=1)
    return

