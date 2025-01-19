import anacal
import galsim
import numpy as np


def test_ngmix_gaussian_fit():
    nx = 32
    ny = 32
    scale = 0.2
    psf_fwhm = 0.7
    sigma_arcsec = 0.52
    klim = 100.0
    # PSF
    psf_obj = galsim.Moffat(
        beta=2.5,
        fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.02,
    )
    psf_array = psf_obj.shift(0.5* scale,0.5 * scale).drawImage(
        nx=nx, ny=ny, scale=scale,
    ).array


    obj = galsim.Exponential(half_light_radius=0.45).shear(g1=0.03)
    obj = galsim.Convolve(psf_obj, obj)
    img_array = obj.shift(0.6 * scale, 0.4 * scale).drawImage(
        nx=nx, ny=ny, scale=scale,
    ).array

    fitter = anacal.ngmix.GaussFit(
        nx=nx, ny=ny, scale=scale, sigma_arcsec=sigma_arcsec, klim=klim,
    )

    # initialize parameters
    params0 = [
        anacal.math.qnumber(1.0, 0.0, 0.0, 0.0, 0.0),  # A
        anacal.math.qnumber(1.0, 0.0, 0.0, 0.0, 0.0),  # rho
        anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),  # G1
        anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),  # G2
        anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),  # x0
        anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),  # y0
    ]

    fitter.run(
        params0=params0,
        xcen=nx//2,
        ycen=ny//2,
        img_array=img_array,
        psf_array=psf_array,
        num_epochs=15,
    )
    e1 = fitter.model.Gamma1
    e2 = fitter.model.Gamma2
    assert abs(e1.v / e1.g1 / 0.03 - 1.0) < 0.003
    assert abs(e2.v / e2.g2) < 1e-4
    return
