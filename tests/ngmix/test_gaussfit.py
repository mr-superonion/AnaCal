import anacal
import galsim
import numpy as np


def test_ngmix_gaussian():
    nx = 32
    ny = 32
    scale = 0.2
    sigma_arcsec = 0.52
    klim = 100.0
    # No PSF
    psf_array = np.zeros((ny, nx))
    psf_array[ny//2, nx//2] = 1

    obj = galsim.Exponential(half_light_radius=0.45).shear(g1=0.03)
    img_array = obj.shift(0.6 * scale, 0.4 * scale).drawImage(
        nx=nx, ny=ny, scale=scale, method="no_pixel"
    ).array

    fitter = anacal.ngmix.Fitting(
        nx=nx, ny=ny, scale=scale, sigma_arcsec=sigma_arcsec, klim=klim,
    )

    # initialize parameters
    params0=[
        anacal.math.qnumber(1.0, 0.0, 0.0, 0.0, 0.0),  # A
        anacal.math.qnumber(1.0, 0.0, 0.0, 0.0, 0.0),  # rho
        anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),  # G1
        anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),  # G2
        anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),  # x0
        anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),  # y0
    ]

    out = fitter.run(
        params0=params0,
        xcen=nx//2,
        ycen=ny//2,
        img_array=img_array,
        psf_array=psf_array,
        num_epochs=20
    )
    assert abs(out[2].v / out[2].g1 / 0.03 - 1.0) < 0.003
    assert abs(out[3].v / out[3].g2) < 1e-4
    return
