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


def test_ngmix_gaussian_fit1():

    prior = anacal.ngmix.modelPrior()
    prior.mu_t = anacal.math.tnumber(np.log(0.6))
    prior.set_sigma_t(anacal.math.tnumber(10.0))
    prior.set_sigma_e(anacal.math.tnumber(100))

    obj2 = galsim.Gaussian(sigma=0.60).shear(g1=0.03)
    obj2 = galsim.Convolve(psf_obj, obj2)
    img_array = obj2.shift((0.5 - 0.1) * scale, (0.5 + 0.1) * scale).drawImage(
        nx=nx, ny=ny, scale=scale,
    ).array

    fitter = anacal.ngmix.GaussFit(
        scale=scale, sigma_arcsec=sigma_arcsec,
    )

    # initialize parameters
    src = anacal.table.galNumber()
    src.model.x1.v = 32 * scale
    src.model.x2.v = 32 * scale
    catalog = [src]
    result = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=30,
        variance=1.0,
    )
    e1 = result[0].model.e1
    e2 = result[0].model.e2
    np.testing.assert_allclose(
        result[0].model.x1.v / scale - 32,
        -0.1, atol=1e-6
    )
    np.testing.assert_allclose(
        result[0].model.x2.v / scale - 32,
        0.1, atol=1e-6
    )
    np.testing.assert_allclose(
        result[0].model.A.v,
        1.0, atol=1e-3, rtol=1e-3,
    )
    np.testing.assert_allclose(
        result[0].model.get_flux_stamp(128, 128, 0.1, 0.6).v,
        1.0, atol=1e-3, rtol=1e-3,
    )
    np.testing.assert_allclose(
        result[0].model.get_flux_stamp(64, 64, 0.2, sigma_arcsec).v,
        1.0, atol=1e-3, rtol=1e-3,
    )
    assert abs(e1.v / e1.g1 / 0.03 - 1.0) < 0.003
    assert abs(e2.v / e2.g2) < 1e-4

    img_obj = anacal.image.ImageT(
        nx=nx, ny=ny,
        scale=scale, sigma_arcsec=sigma_arcsec,
        klim=klim,
    )

    img_array1 = img_obj.prepare_tnumber_image(
        img_array,
        psf_array,
    )
    img_array2 = result[0].model.get_image_stamp(nx, ny, scale, sigma_arcsec)
    diff = img_array1 - img_array2
    assert np.sum(np.abs(diff[1])) / np.sum(np.abs(img_array1)) < 0.1
    return


def test_ngmix_gaussian_fit4():
    prior = anacal.ngmix.modelPrior()
    prior.mu_t = anacal.math.tnumber(-0.5)
    prior.set_sigma_t(anacal.math.tnumber(4.0))
    prior.set_sigma_e(anacal.math.tnumber(25))
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
        scale=scale, sigma_arcsec=sigma_arcsec
    )

    # initialize parameters
    catalog = []
    for center in centers:
        src = anacal.table.galNumber()
        src.model.A.v = 0.5
        src.model.x1.v = center[0] * scale
        src.model.x2.v = center[1] * scale
        catalog.append(src)

    img_array = img_array
    result = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=30,
        variance=1.0,
    )
    for rr in result:
        e1 = rr.model.e1
        e2 = rr.model.e2
        np.testing.assert_allclose(
            rr.model.x1.g1 - (rr.model.x1.v - nx / 2 * scale),
            0.0, rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            rr.model.x1.g2 - (rr.model.x2.v - ny / 2 * scale),
            0.0, rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            rr.model.x2.g2 - (rr.model.x1.v - nx / 2 * scale),
            0.0, rtol=1e-6, atol=1e-6,
        )
        np.testing.assert_allclose(
            rr.model.x2.g1 + (rr.model.x2.v - ny / 2 * scale),
            0.0, rtol=1e-6, atol=1e-6,
        )
        assert abs(e1.v / e1.g1 / 0.03 - 1.0) < 0.003
        assert abs(e2.v / e2.g2) < 1e-4
    return
