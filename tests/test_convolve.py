import anacal
import numpy as np

from .sim import gauss_kernel_rfft, gauss_tophat_kernel_rfft


def test_ini_final(ny=128, nx=64):
    Conv = anacal.convolve.Convolve()
    ny = 64
    nx = 64
    data = np.zeros((ny, nx))
    data[ny//2, nx//2] = 1.0
    Conv.initialize(data, 1.0)
    data2 = Conv.finalize()
    np.testing.assert_almost_equal(data, data2)
    return


if __name__ == "__main__":
    test_ini_final()
