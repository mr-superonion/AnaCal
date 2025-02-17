import anacal
import galsim
import numpy as np


nx = 64
ny = 64
scale = 0.2
psf_fwhm = 0.7
sigma_arcsec = 0.4
klim = 100.0
# PSF
psf_obj = galsim.Moffat(
    beta=2.5,
    fwhm=psf_fwhm,
).shear(
    g1=0.02,
    g2=-0.02,
)
psf_array = psf_obj.shift(0.5 * scale, 0.5 * scale).drawImage(
    nx=nx, ny=ny, scale=scale,
).array

obj = galsim.Exponential(half_light_radius=0.45).shear(g1=0.03)
obj = galsim.Convolve(psf_obj, obj)

prior_mu = anacal.ngmix.modelNumber(
    anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),   # A
    anacal.math.qnumber(-0.5, 0.0, 0.0, 0.0, 0.0),  # t
    anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),   # e1
    anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),   # e2
    anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),   # x0
    anacal.math.qnumber(0.0, 0.0, 0.0, 0.0, 0.0),   # y0
)

prior_sigma = anacal.ngmix.modelNumber(
    anacal.math.qnumber(-1.0, 0.0, 0.0, 0.0, 0.0),  # A
    anacal.math.qnumber(4.0, 0.0, 0.0, 0.0, 0.0),   # t
    anacal.math.qnumber(2.5, 0.0, 0.0, 0.0, 0.0),   # e1
    anacal.math.qnumber(2.5, 0.0, 0.0, 0.0, 0.0),   # e2
    anacal.math.qnumber(-1.0, 0.0, 0.0, 0.0, 0.0),  # x0
    anacal.math.qnumber(-1.0, 0.0, 0.0, 0.0, 0.0),  # y0
)


def test_ngmix_gaussian_fit1():
    obj2 = galsim.Gaussian(sigma=0.60).shear(g1=0.03)
    obj2 = galsim.Convolve(psf_obj, obj2)
    img_array = obj2.shift((0.5 - 0.1) * scale, (0.5 + 0.1) * scale).drawImage(
        nx=nx, ny=ny, scale=scale,
    ).array

    fitter = anacal.ngmix.GaussFit(
        nx=nx, ny=ny, scale=scale, sigma_arcsec=sigma_arcsec, klim=klim,
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
    )

    # initialize parameters
    src = anacal.ngmix.galNumber()
    src.params.x1.v = 32 * scale
    src.params.x2.v = 32 * scale
    catalog = [src]
    img_array = img_array / scale ** 2.0
    result = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        num_epochs=20,
        variance=1.0 / scale ** 4.0,
    )
    e1 = result[0].params.e1
    e2 = result[0].params.e2
    np.testing.assert_allclose(
        result[0].params.x1.v / scale - 32,
        -0.1, atol=1e-6
    )
    np.testing.assert_allclose(
        result[0].params.x2.v / scale - 32,
        0.1, atol=1e-6
    )
    np.testing.assert_allclose(
        fitter.model.get_flux_stamp(128, 128, 0.1).v,
        1.0, atol=1e-2, rtol=1e-2
    )
    assert abs(e1.v / e1.g1 / 0.03 - 1.0) < 0.003
    assert abs(e2.v / e2.g2) < 1e-4
    return


def test_ngmix_gaussian_fit4():
    nx = 256
    ny = 64

    # Create an empty image
    full_image = galsim.ImageF(ncol=nx, nrow=ny, scale=scale)

    # Define centers for 4 substamps
    centers = [(30, 30), (96, 32), (160, 32), (224, 32)]

    # Draw galaxies at specified positions
    for center in centers:
        shift = galsim.PositionD(
            (center[0] - nx / 2) * scale,
            (center[1] - ny / 2) * scale
        )
        final_galaxy = obj.shift(shift)
        final_galaxy.drawImage(image=full_image, add_to_image=True)
    img_array = full_image.array

    fitter = anacal.ngmix.GaussFit(
        nx=nx, ny=ny, scale=scale, sigma_arcsec=sigma_arcsec, klim=klim,
        prior_mu=prior_mu,
        prior_sigma=prior_sigma,
    )

    # initialize parameters
    catalog = []
    for center in centers:
        src = anacal.ngmix.galNumber()
        src.params.x1.v = center[0] * scale
        src.params.x2.v = center[1] * scale
        src.params.A.v = 0.48
        catalog.append(src)

    img_array = img_array / scale ** 2.0
    result = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        num_epochs=10,
        variance=1.0 / scale ** 4.0,
    )
    for i in range(4):
        e1 = result[i].params.e1
        e2 = result[i].params.e2
        assert abs(e1.v / e1.g1 / 0.03 - 1.0) < 0.003
        assert abs(e2.v / e2.g2) < 1e-4
    return
