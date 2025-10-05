import os

import anacal
import fitsio
import galsim
import numpy as np


def test_noise_covariance():
    variance = 0.22**2.0 / 2.0
    sigma_as = 0.55
    pixel_scale = 0.2
    ngrid = 64
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0)
    psf_obj = psf_obj.shear(e1=0.02, e2=-0.02)
    psf_array = (
        psf_obj.shift(0.5 * pixel_scale, 0.5 * pixel_scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=pixel_scale)
        .array
    )
    test_fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cov_elem.fits",
    )
    cov_elem = fitsio.read(test_fname)

    ftask = anacal.fpfs.FpfsTask(
        npix=psf_array.shape[0],
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        sigma_shapelets=sigma_as,
        do_detection=True,
        kmax_thres=1e-20,
        noise_variance=variance,
    )
    cov_elem2 = ftask.prepare_covariance(variance=variance)
    np.testing.assert_allclose(cov_elem, cov_elem2, atol=1e-6, rtol=0)
    np.testing.assert_allclose(np.diag(cov_elem), np.diag(cov_elem2), rtol=1e-6)
    np.testing.assert_allclose(
        np.sqrt(np.diagonal(cov_elem)),
        ftask.std_modes,
    )
    return
