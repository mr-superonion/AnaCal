import anacal
import fpfs
import galsim
import numpy as np
import numpy.lib.recfunctions as rfn

from .fpfs import smooth

scale = 0.2
ngrid = 128
psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
    e1=0.02, e2=-0.02
)
psf_array = (
    psf_obj.shift(0.5 * scale, 0.5 * scale)
    .drawImage(nx=32, ny=32, scale=scale)
    .array
)

nord = 4
det_nrot = 4
sigma_as = 0.53

task = fpfs.image.measure_source(
    psf_array,
    pix_scale=scale,
    sigma_arcsec=sigma_as,
    nord=nord,
    det_nrot=det_nrot,
)

gal_obj = (
    psf_obj.shift(-3.5, 2) * 2
    + psf_obj.shift(2, -1) * 4
    + psf_obj.shift(-2, -0.5) * 4
    + psf_obj.shift(-3.2, 0.5) * 6
)
gal_data = gal_obj.drawImage(nx=ngrid, ny=ngrid, scale=scale).array


def test_convolve():
    det_task = anacal.fpfs.FpfsImage(
        nx=ngrid,
        ny=ngrid,
        scale=scale,
        sigma_arcsec=sigma_as,
        klim=task.klim / scale,
        psf_array=psf_array,
    )
    smooth_data = det_task.smooth_image(gal_array=gal_data, noise_array=None)
    smooth_data2 = smooth(task, gal_data, psf_array)
    np.testing.assert_almost_equal(smooth_data, smooth_data2)
    return


def test_convolve_noise(seed=2):
    np.random.seed(seed=seed)
    det_task = anacal.fpfs.FpfsImage(
        nx=ngrid,
        ny=ngrid,
        scale=scale,
        sigma_arcsec=sigma_as,
        klim=task.klim / scale,
        psf_array=psf_array,
    )

    noise_array = np.random.randn(ngrid, ngrid)
    smooth_data = det_task.smooth_image(
        gal_array=gal_data, noise_array=noise_array
    )
    smooth_data2 = smooth(task, gal_data, psf_array, noise_array)
    np.testing.assert_almost_equal(smooth_data, smooth_data2)
    return


def test_detect():
    std = 0.4

    cov_element = np.ones((task.ncol, task.ncol)) * std**2.0
    det_task = anacal.fpfs.FpfsImage(
        nx=ngrid,
        ny=ngrid,
        scale=scale,
        sigma_arcsec=sigma_as,
        klim=task.klim / scale,
        psf_array=psf_array,
    )
    noise_array = np.random.randn(ngrid, ngrid)
    out1 = det_task.detect_source(
        gal_array=gal_data,
        fthres=1.0,
        pthres=anacal.fpfs.fpfs_det_sigma2 + 0.02,  # effectively v>0
        bound=2,
        std_m00=std * scale**2.0,
        std_v=std * scale**2.0,
        noise_array=noise_array,
    )
    out1 = rfn.structured_to_unstructured(out1)
    out1 = out1[:, :-1]

    out2 = task.detect_source(
        gal_data,
        psf_array,
        cov_element,
        fthres=1.0,
        pthres=0.8,
        pratio=0.0,
        bound=2,
        noise_array=noise_array,
    )
    assert out1.shape == out2.shape
    np.testing.assert_almost_equal(out1, out2)
    return


if __name__ == "__main__":
    test_convolve()
    test_convolve_noise()
    test_detect()
