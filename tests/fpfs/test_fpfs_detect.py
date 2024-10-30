import anacal
import galsim
import numpy as np
import numpy.lib.recfunctions as rfn

from . import smooth

scale = 0.2
ngrid = 1000
psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
    e1=0.02, e2=-0.02
)
psf_array = (
    psf_obj.shift(0.5 * scale, 0.5 * scale)
    .drawImage(nx=32, ny=32, scale=scale)
    .array
)

sigma_arcsec = 0.53

gal_obj = (
    psf_obj.shift(-3.5, 2) * 2
    + psf_obj.shift(2, -1) * 4
    + psf_obj.shift(-2, -0.5) * 4
    + psf_obj.shift(-3.2, 0.5) * 6
)
gal_data = gal_obj.drawImage(nx=ngrid, ny=ngrid, scale=scale).array
ncol = 21
kmax = 2.945243112740431
sigmaf = 0.37735849056603776
klim = kmax / scale


def test_convolve_image():
    det_task = anacal.fpfs.FpfsImage(
        nx=ngrid,
        ny=ngrid,
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        klim=klim,
        psf_array=psf_array,
    )
    smooth_data = det_task.smooth_image(gal_array=gal_data, noise_array=None)
    smooth_data2 = smooth(sigmaf, kmax, gal_data, psf_array)
    np.testing.assert_almost_equal(smooth_data, smooth_data2)
    return


def test_convolve_noise(seed=2):
    np.random.seed(seed=seed)
    det_task = anacal.fpfs.FpfsImage(
        nx=ngrid,
        ny=ngrid,
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        klim=klim,
        psf_array=psf_array,
    )

    noise_array = np.random.randn(ngrid, ngrid)
    smooth_data = det_task.smooth_image(
        gal_array=gal_data, noise_array=noise_array
    )
    smooth_data2 = smooth(sigmaf, kmax, gal_data, psf_array, noise_array)
    np.testing.assert_almost_equal(smooth_data, smooth_data2)
    return


def test_detect():
    std = 0.4

    cov_element = np.ones((ncol, ncol)) * std**2.0
    det_task = anacal.fpfs.FpfsImage(
        nx=150,
        ny=200,
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        klim=klim,
        psf_array=psf_array,
        npix_overlap=64,
        bound=20,
    )
    noise_array = np.random.randn(ngrid, ngrid)
    out1 = det_task.detect_source(
        gal_array=gal_data + noise_array,
        fthres=1.0,
        pthres=anacal.fpfs.fpfs_det_sigma2 + 0.02,  # effectively v>0
        std_m00=std * scale**2.0,
        omega_v=std * scale**2.0 * 1.6,
        v_min=std * scale**2.0 * 0.8,
    )
    out1 = rfn.structured_to_unstructured(out1)
    out1 = out1[:, :-1]
    out1 = out1[np.lexsort((out1[:, 0], out1[:, 1]))]
    assert out1.shape == (48666, 3)
    arr = np.array(
        [
            [50, 21, 0],
            [67, 21, 0],
            [111, 21, 1],
        ]
    )
    np.testing.assert_almost_equal(out1[:3], arr)
    return


if __name__ == "__main__":
    test_convolve_image()
    test_convolve_noise()
    test_detect()
