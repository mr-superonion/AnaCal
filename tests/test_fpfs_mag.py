import anacal
import galsim
import numpy as np


def test_catalog():

    scale = 0.2
    psf_obj = galsim.Moffat(fwhm=0.8, beta=2.5)
    ngrid = 256
    psf_data = psf_obj.shift(scale * 0.5, scale * 0.5).drawImage(
        nx=64, ny=64, scale=scale, method="no_pixel",
    ).array
    gal_obj = galsim.Convolve([galsim.Gaussian(sigma=0.52, flux=100), psf_obj])
    gal_data = gal_obj.shift(scale * 0.5, scale*0.5).drawImage(
        nx=ngrid, ny=ngrid, scale=scale, method="no_pixel",
    ).array

    pthres = 0.2
    pratio = 0.0
    std = 1
    det_nrot = 4
    nord=4
    bound=0
    pixel_scale=0.2
    cov_element = np.ones((21, 21)) * std**2.0 * pixel_scale**4.0
    sigma_as = 0.52
    mag_zero = 30.0
    dtask = anacal.fpfs.FpfsDetect(
        nx=gal_data.shape[1],
        ny=gal_data.shape[0],
        psf_array=psf_data,
        pixel_scale=scale,
        sigma_arcsec=sigma_as,
        cov_matrix=cov_element,
        det_nrot=det_nrot,
    )
    mtask = anacal.fpfs.FpfsMeasure(
        psf_array=psf_data,
        pixel_scale=scale,
        sigma_arcsec=sigma_as,
        nord=nord,
        det_nrot=det_nrot,
    )
    det = dtask.run(
        gal_array=gal_data,
        fthres=1.0,
        pthres=pthres,
        pratio=pratio,
        pthres2=anacal.fpfs.fpfs_det_sigma2 + 0.02,
        bound=bound,
        noise_array=None,
    )
    src = mtask.run(gal_array=gal_data, det=det)

    cat_obj = anacal.fpfs.FpfsCatalog(
        pixel_scale=pixel_scale,
        sigma_arcsec=sigma_as,
        mag_zero=mag_zero,
        cov_matrix=cov_element,
        snr_min=12,
        r2_min=0.1,
        c0=4,
        c2=10,
        alpha=1.0,
        beta=0.0,
        pthres=pthres,
        pratio=pratio,
        pthres2=0.12,
        det_nrot=det_nrot,
    )
    mag1 = cat_obj.measure_mag(src)
    mag2 = mag_zero - np.log10(np.sum(gal_data)) * 2.5
    assert (mag1 - mag2) < 0.01

    return
