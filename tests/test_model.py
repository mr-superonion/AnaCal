import anacal
import numpy as np

from .sim import gauss_kernel_rfft, gauss_tophat_kernel_rfft


def test_gausstophat(
    d=np.pi * 0.65,
    sigma=1.0 / 3.0,
    theta=np.pi / 20,
    gamma1=0.1,
    gamma2=-0.3,
):
    ny=64; nx=64
    gt_model = anacal.model.GaussianTopHat(d=d, sigma=sigma)
    gt_model.set_transform(theta=theta, gamma1=gamma1, gamma2=gamma2)
    img = gt_model.draw(scale=1, nx=nx, ny=ny)
    np.testing.assert_almost_equal(img[0, 0], 1.0)

    img2 = gauss_tophat_kernel_rfft(ny, nx, d, sigma, theta, gamma1, gamma2)
    np.testing.assert_almost_equal(img, img2)
    return

def test_gauss(
    sigma=1.0 / 3.0,
    theta=np.pi / 20,
    gamma1=0.1,
    gamma2=-0.3,
):
    ny=64; nx=128
    gt_model = anacal.model.Gaussian(sigma=sigma)
    gt_model.set_transform(theta=theta, gamma1=gamma1, gamma2=gamma2)
    img = gt_model.draw(scale=1, nx=nx, ny=ny)
    np.testing.assert_almost_equal(img[0, 0], 1.0)

    img2 = gauss_kernel_rfft(ny, nx, sigma, 10.0, theta, gamma1, gamma2)
    np.testing.assert_almost_equal(img, img2)
    return

if __name__ == "__main__":
    test_gausstophat()
    test_gauss()
