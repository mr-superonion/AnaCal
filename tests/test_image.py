import gc
import time

import anacal
import galsim
import numpy as np

import fpfs

from . import mem_used, print_mem
from .sim import gauss_tophat_kernel_rfft


def test_set_r(ny=122, nx=64):
    img_obj = anacal.image.Image(nx=nx, ny=ny, scale=1.0, use_estimate=True)
    data = np.zeros((32, 32))
    data[16, 16] = 1.0
    img_obj.set_r(data)
    img1 = img_obj.draw_r()
    assert img1[ny // 2, nx // 2] == 1
    assert np.sum(img1) == 1
    img_obj.set_r(data, ishift=True)
    img2 = img_obj.draw_r()
    assert img2[0, 0] == 1
    assert np.sum(img2) == 1
    return


def test_fft(ny=128, nx=64):
    img_obj = anacal.image.Image(nx=nx, ny=ny, scale=1.0)
    data = np.zeros((ny, nx))
    data[ny // 2, nx // 2] = 1.0
    img_obj.set_r(data)
    img_obj.fft()
    img_obj.ifft()
    data2 = img_obj.draw_r(False)
    np.testing.assert_almost_equal(data, data2)
    data3 = img_obj.draw_r(True)
    np.testing.assert_almost_equal(np.fft.fftshift(data2), data3)
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
    img_obj = anacal.image.Image(nx=nx, ny=ny, scale=1.0)
    data = np.zeros((ny, nx))
    data[0, 0] = 1.0
    img_obj.set_r(data)
    img_obj.fft()
    gt_model = anacal.model.GaussianTopHat(d=d, sigma=sigma)
    gt_model.set_transform(theta=theta, gamma1=gamma1, gamma2=gamma2)
    img_obj.filter(filter_model=gt_model)
    img = img_obj.draw_f()
    img2 = gauss_tophat_kernel_rfft(ny, nx, d, sigma, theta, gamma1, gamma2)
    np.testing.assert_almost_equal(img, img2)

    img2 = np.fft.irfft2(img2, (ny, nx))
    img_obj.ifft()
    img = img_obj.draw_r()
    np.testing.assert_almost_equal(img, img2)
    return


def test_deconvolve_image(seed=1):
    rcut = 32
    scale = 0.2
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )
    ngrid = 64
    nrot = 1

    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )
    psf_data = psf_data[
        ngrid // 2 - rcut : ngrid // 2 + rcut,
        ngrid // 2 - rcut : ngrid // 2 + rcut,
    ]
    gname = "g1-0"
    gal_data = anacal.simulation.make_isolated_sim(
        gal_type="mixed",
        sim_method="fft",
        psf_obj=psf_obj,
        gname=gname,
        seed=seed,
        ny=ngrid,
        nx=ngrid * nrot,
        scale=scale,
        do_shift=False,
        buff=0,
        nrot_per_gal=nrot,
    )[0]

    indx = np.arange(ngrid // 2, ngrid * nrot, ngrid)
    indy = np.arange(ngrid // 2, ngrid, ngrid)
    inds = np.meshgrid(indy, indx, indexing="ij")
    coords = np.vstack(inds).T
    norder = 4
    det_nrot = 4
    # test shear estimation
    task = fpfs.image.measure_source(
        psf_data,
        pix_scale=scale,
        sigma_arcsec=0.53,
        nord=norder,
        det_nrot=det_nrot,
    )
    # linear observables
    mms = task.measure(gal_data, coords)

    dec_obj = anacal.image.Image(nx=ngrid, ny=ngrid, scale=1.0)
    dec_obj.set_r(psf_data, ishift=True)
    dec_obj.fft()

    img_obj = anacal.image.Image(nx=ngrid, ny=ngrid, scale=1.0)
    img_obj.set_r(gal_data)
    img_obj.fft()
    gt_model = anacal.model.Gaussian(sigma=task.sigmaf)
    img_obj.filter(filter_model=gt_model)
    img_obj.deconvolve(psf_image=dec_obj.draw_f(), klim=task.klim)
    img_obj.ifft()
    img = img_obj.draw_r()
    obs = img[ngrid // 2, ngrid // 2] / scale**2.0
    np.testing.assert_almost_equal(obs, mms[0, 0], decimal=4)
    return


def test_rotate90():
    scale = 0.2
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=3.0).shear(
        e1=0.1, e2=-0.02
    )
    ngrid = 64

    psf_data = (
        psf_obj.shift(4.5 * scale, 17 * scale)
        .shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )
    psf_data2 = (
        psf_obj.shift(4.5 * scale, 17 * scale)
        .rotate(90 * galsim.degrees)
        .shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )
    imobj = anacal.image.Image(nx=ngrid, ny=ngrid, scale=1.0)
    imobj.set_r(psf_data)
    imobj.fft()
    imobj.rotate90_f()
    imobj.ifft()
    psf_rot = imobj.draw_r()
    np.testing.assert_almost_equal(psf_rot, psf_data2)
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
    gc.collect()

    initial_memory_usage = mem_used()
    print_mem(initial_memory_usage)
    t0 = time.time()
    for _ in range(3):
        img_obj = anacal.image.Image(nx=nx, ny=ny, scale=1.0)
        data = np.zeros((ny, nx), dtype=np.float64)
        data[0, 0] = np.random.random(1)
        img_obj.set_r(data)
        img_obj.fft()
        gt_model = anacal.model.GaussianTopHat(d=d, sigma=sigma)
        gt_model.set_transform(theta=theta, gamma1=gamma1, gamma2=gamma2)
        img_obj.filter(filter_model=gt_model)
        img = img_obj.draw_f()
        img_obj.ifft()
        print_mem(mem_used() - initial_memory_usage)
        del img_obj, img, gt_model, data
    t1 = time.time()
    print("Time: ", t1 - t0)
    gc.collect()
    final_memory_usage = mem_used()
    print_mem(final_memory_usage - initial_memory_usage)
    assert final_memory_usage < 1.2 * initial_memory_usage
    return


if __name__ == "__main__":
    test_fft()
    test_gausstophat()
    test_memory()
    test_set_r()
