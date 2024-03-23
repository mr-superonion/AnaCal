import gc

import anacal
import numpy as np

from tests import mem_used

from . import mem_used, print_mem
from .sim import gauss_tophat_kernel_rfft


def test_fft(ny=128, nx=64):
    img_obj = anacal.image.Image()
    data = np.zeros((ny, nx))
    data[ny // 2, nx // 2] = 1.0
    img_obj.initialize(nx=nx, ny=ny, scale=1.0)
    img_obj.set_r(data)
    img_obj.fft()
    img_obj.ifft()
    data2 =  img_obj.draw_r()
    np.testing.assert_almost_equal(data, data2)
    return


def test_gausstophat(
    d=np.pi * 0.65,
    sigma=1.0 / 3.0,
    theta=np.pi / 20,
    gamma1=0.1,
    gamma2=-0.3,
):
    ny = 64
    nx = 256
    gt_model = anacal.model.GaussianTopHat(d=d, sigma=sigma)
    gt_model.set_transform(theta=theta, gamma1=gamma1, gamma2=gamma2)

    img_obj = anacal.image.Image()
    data = np.zeros((ny, nx))
    data[0, 0] = 1.0
    img_obj.initialize(nx=nx, ny=ny, scale=1.0)
    img_obj.set_r(data)
    img_obj.fft()
    gt_model = anacal.model.GaussianTopHat(d=d, sigma=sigma)
    gt_model.set_transform(theta=theta, gamma1=gamma1, gamma2=gamma2)
    img_obj.filter(gt_model)
    img = img_obj.draw_f()
    img2 = gauss_tophat_kernel_rfft(ny, nx, d, sigma, theta, gamma1, gamma2)
    np.testing.assert_almost_equal(img, img2)

    img2 = np.fft.irfft2(img2, (ny, nx))
    img_obj.ifft()
    img = img_obj.draw_r()
    np.testing.assert_almost_equal(img, img2)
    return


def test_memory(
    d=np.pi * 0.65,
    sigma=1.0 / 3.0,
    theta=np.pi / 20,
    gamma1=0.1,
    gamma2=-0.3,
):
    print()
    ny = 4096
    nx = 4096
    mem_expected_r = nx * ny * np.float64().itemsize * 2
    mem_expected_i = nx * (ny / 2 + 1) * np.complex128().itemsize
    gc.collect()

    initial_memory_usage = mem_used()
    print_mem(initial_memory_usage)
    for _ in range(10):
        gt_model = anacal.model.GaussianTopHat(d=d, sigma=sigma)
        gt_model.set_transform(theta=theta, gamma1=gamma1, gamma2=gamma2)
        img_obj = anacal.image.Image()
        data = np.zeros((ny, nx), dtype=np.float64)
        data[0, 0] = np.random.random(1)
        img_obj.initialize(nx=nx, ny=ny, scale=1.0)
        img_obj.set_r(data)
        img_obj.fft()
        gt_model = anacal.model.GaussianTopHat(d=d, sigma=sigma)
        gt_model.set_transform(theta=theta, gamma1=gamma1, gamma2=gamma2)
        img_obj.filter(gt_model)
        img = img_obj.draw_f()
        print_mem(mem_used() - initial_memory_usage)
        del img_obj, img, gt_model, data
    gc.collect()
    final_memory_usage = mem_used()
    print_mem(final_memory_usage - initial_memory_usage)
    assert final_memory_usage < 1.2 * initial_memory_usage
    return


if __name__ == "__main__":
    test_fft()
    test_gausstophat()
    test_memory()
