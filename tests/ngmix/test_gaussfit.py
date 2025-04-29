import anacal
import galsim
import numpy as np


def test_ngmix_gaussian_fit_additive(test_g1=True):
    nx = 64
    ny = 64
    scale = 0.2
    psf_fwhm = 0.7
    sigma_arcsec = 0.4
    dx1 = -0.2
    dx2 = 0.11
    # PSF
    psf_obj = galsim.Moffat(
        beta=2.5,
        fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.02,
    )
    psf_array = (
        psf_obj.shift(0.5 * scale, 0.5 * scale).drawImage(
            nx=nx,
            ny=ny,
            scale=scale,
        )
        .array
    )

    obj0 = galsim.Exponential(half_light_radius=0.2)

    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        stamp_size=32,
    )

    def make_sim(g1, g2, angle=0.0, flux=1.0):
        obj = obj0.rotate(angle * galsim.degrees).withFlux(flux)
        obj = obj.shear(g1=g1, g2=g2)
        obj = obj.shift((0.5 + dx1) * scale, (0.5 + dx2) * scale)
        obj = galsim.Convolve(psf_obj, obj)

        # Create an empty image
        full_image = galsim.ImageF(ncol=nx, nrow=ny, scale=scale)

        # Define centers for 100 substamps
        crange = np.arange(32, 64, 64)
        centers = [(x, y) for x in crange for y in crange]

        # Draw galaxies at specified positions
        for center in centers:
            shift = galsim.PositionD(
                (center[0] - nx / 2) * scale, (center[1] - ny / 2) * scale
            )
            final_galaxy = obj.shift(shift)
            final_galaxy.drawImage(image=full_image, add_to_image=True)
        return full_image.array

    flux = 150.0
    num_epochs = 35
    src = anacal.table.galNumber()
    src.model.x1.v = 32 * scale
    src.model.x2.v = 32 * scale
    src.model.F.v = 1.0
    src.model.t.v = -0.5
    catalog = [src]
    prior = anacal.ngmix.modelPrior()

    img_array = make_sim(g1=0, g2=0, flux=flux)
    cat_1 = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]
    ell1 = cat_1.model.get_shape()[0]
    assert np.abs(ell1.v / ell1.g1)< 1e-4
    return


def test_ngmix_gaussian_fit2():
    nx = 64
    ny = 64
    scale = 0.2
    psf_fwhm = 0.7
    sigma_arcsec = 0.4
    dx1 = -0.2
    dx2 = 0.11

    # PSF
    psf_obj = galsim.Moffat(
        beta=2.5,
        fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.02,
    )
    psf_array = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(
            nx=nx,
            ny=ny,
            scale=scale,
        )
        .array
    )


    obj0 = galsim.Exponential(half_light_radius=0.2)
    obj0 = obj0.shear(e1=0.2, e2=-0.1)

    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        stamp_size=32,
    )

    def make_sim(g1, g2, angle=0.0, flux=1.0):
        obj = obj0.rotate(angle * galsim.degrees).withFlux(flux)
        obj = obj.shear(g1=g1, g2=g2)
        obj = obj.shift((0.5 + dx1) * scale, (0.5 + dx2) * scale)
        obj = galsim.Convolve(psf_obj, obj)

        # Create an empty image
        full_image = galsim.ImageF(ncol=nx, nrow=ny, scale=scale)

        # Define centers for 100 substamps
        crange = np.arange(nx // 2, nx, nx)
        centers = [(x, y) for x in crange for y in crange]

        # Draw galaxies at specified positions
        for center in centers:
            shift = galsim.PositionD(
                (center[0] - nx / 2) * scale, (center[1] - ny / 2) * scale
            )
            final_galaxy = obj.shift(shift)
            final_galaxy.drawImage(image=full_image, add_to_image=True)
        return full_image.array

    flux = 150.0
    num_epochs = 30
    src = anacal.table.galNumber()
    src.model.x1.v = nx // 2 * scale
    src.model.x2.v = nx // 2 * scale
    src.model.F.v = 1.0
    catalog = [src]
    prior = anacal.ngmix.modelPrior()

    g1_list = [0.02, -0.02]
    g2_list = [0.0, 0.0]

    # Test shear response calculation (no multiplicative bias)
    img_array = make_sim(g1=g1_list[0], g2=g2_list[0], flux=flux)
    cat_1 = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]
    ell1 = cat_1.model.get_shape()[0]

    assert np.abs((cat_1.model.F.v - flux) / flux) < 0.01
    np.testing.assert_allclose(
        cat_1.model.x1.v / scale - nx // 2, dx1,
        atol=1e-5, rtol=0,
    )
    np.testing.assert_allclose(
        cat_1.model.x2.v / scale - ny // 2, dx2,
        atol=1e-5, rtol=0,
    )
    np.testing.assert_allclose(
        cat_1.model.get_flux_stamp(128, 128, 0.1, 0.6).v,
        flux,
        atol=0,
        rtol=1e-2,
    )
    np.testing.assert_allclose(
        cat_1.model.get_flux_stamp(64, 64, 0.2, sigma_arcsec).v,
        flux,
        atol=0,
        rtol=1e-2,
    )

    img_array = make_sim(g1=g1_list[1], g2=g2_list[1], flux=flux*2)
    cat_2 = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=1.0,
    )[0]
    ell2 = cat_2.model.get_shape()[0]

    assert np.abs(
        (ell1.v - ell2.v) / (ell2.g1 + ell1.g1) * 2.0 / 0.04 - 1
    ) < 2e-3

    klim = 100.0
    img_obj = anacal.image.ImageQ(
        nx=nx,
        ny=ny,
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        klim=klim,
    )

    img_array1 = img_obj.prepare_qnumber_image(
        img_array,
        psf_array,
    )
    img_array2 = cat_2.model.get_image_stamp(nx, ny, scale, sigma_arcsec)
    diff = img_array1[0:3] - img_array2
    assert np.sum(np.abs(diff[0])) / np.sum(np.abs(img_array1[0])) < 2e-2
    assert np.sum(np.abs(diff[1])) / np.sum(np.abs(img_array1[1])) < 2e-1
    assert np.sum(np.abs(diff[2])) / np.sum(np.abs(img_array1[2])) < 2e-1

    # Test symmetry
    cat_1 = fitter.process_block(
        catalog=catalog,
        img_array=make_sim(g1=0.00, g2=0.0, angle=0.0, flux=flux),
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]
    ell1 = cat_1.model.get_shape()[0]

    cat_2 = fitter.process_block(
        catalog=catalog,
        img_array=make_sim(g1=0.00, g2=0.0, angle=90.0, flux=flux),
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]
    ell2 = cat_2.model.get_shape()[0]
    assert np.abs((ell2.v + ell1.v) / (ell2.g1 + ell1.g1)) < 5e-5

    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        stamp_size=32,
        force_size=True,
    )

    img_array = make_sim(g1=g1_list[0], g2=g2_list[0], flux=flux)
    cat_1 = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]

    ell1 = cat_1.model.get_shape()[0]
    assert ell1.v == 0
    assert ell1.g1 == 0
    assert ell1.g2 == 0

    np.testing.assert_allclose(
        cat_1.model.x1.v / scale - nx // 2, dx1,
        atol=1e-6, rtol=0.0,
    )
    np.testing.assert_allclose(
        cat_1.model.x2.v / scale - ny // 2, dx2,
        atol=1e-6, rtol=0.0,
    )
    img_array = make_sim(g1=g1_list[1], g2=g2_list[1], flux=flux)
    cat_2 = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=1.0,
    )[0]

    ell1 = cat_1.fpfs_e1
    ell2 = cat_2.fpfs_e1
    assert np.abs(
        (ell1.v - ell2.v) / (ell2.g1 + ell1.g1) * 2.0 / 0.04 - 1
    ) < 2e-3

    return


def test_ngmix_gaussian_fit4():
    prior = anacal.ngmix.modelPrior()
    nx = 256
    ny = 64

    scale = 0.2
    psf_fwhm = 0.7
    sigma_arcsec = 0.4
    # PSF
    psf_obj = galsim.Moffat(
        beta=2.5,
        fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.02,
    )
    psf_array = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(
            nx=nx,
            ny=ny,
            scale=scale,
        )
        .array
    )

    obj = galsim.Gaussian(half_light_radius=0.25).shear(g1=0.03)
    obj = galsim.Convolve(psf_obj, obj)

    # Create an empty image
    full_image = galsim.ImageF(ncol=nx, nrow=ny, scale=scale)

    # Define centers for 4 substamps
    centers = [(31.2, 31.2), (95.9, 32.05), (160, 32.1), (224, 31.8)]
    fluxes = [12, 23, 8.5, 18.4]

    # Draw galaxies at specified positions
    for i, center in enumerate(centers):
        shift = galsim.PositionD(
            (center[0] - (nx - 1) / 2) * scale,
            (center[1] - (ny - 1) / 2) * scale
        )
        final_galaxy = obj.shift(shift).withFlux(fluxes[i])
        final_galaxy.drawImage(image=full_image, add_to_image=True)
    img_array = full_image.array

    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        stamp_size=48
    )

    # initialize parameters
    catalog = []
    for center in centers:
        src = anacal.table.galNumber()
        src.model.x1.v = center[0] * scale
        src.model.x2.v = center[1] * scale
        catalog.append(src)

    result = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=25,
        variance=1.0,
    )
    for i, rr in enumerate(result):
        [e1, e2] = rr.model.get_shape()
        np.testing.assert_allclose(
            rr.model.x1.g1 - (rr.model.x1.v - nx / 2 * scale),
            0.0, atol=1e-5, rtol=0,
        )
        np.testing.assert_allclose(
            rr.model.x1.g2 - (rr.model.x2.v - ny / 2 * scale),
            0.0, atol=1e-5, rtol=0,
        )
        np.testing.assert_allclose(
            rr.model.x2.g2 - (rr.model.x1.v - nx / 2 * scale),
            0.0, atol=1e-5, rtol=0,
        )
        np.testing.assert_allclose(
            rr.model.x2.g1 + (rr.model.x2.v - ny / 2 * scale),
            0.0, atol=1e-5, rtol=0,
        )
        assert abs(e1.v / e1.g1 / 0.03 - 1.0) < 0.003
        assert abs(e2.v / e2.g2) < 2e-5
        np.testing.assert_allclose(
            rr.model.F.v,
            fluxes[i],
            atol=1e-1, rtol=1e-2,
        )
    return

# # Loss function
# num_epochs=1
# loss_array = []
# ddloss_array = []
# t_array = np.arange(-5, 5.0, 0.005)
# for t in t_array:
#     src.model.t.v = t
#     # Test shear response calculation (no multiplicative bias)
#     cat_1 = fitter.process_block(
#         catalog=catalog,
#         img_array=make_sim(g1=-0.02, g2=0.0, flux=flux),
#         psf_array=psf_array,
#         prior=prior,
#         num_epochs=num_epochs,
#         variance=1.0,
#     )[0]
#     loss_array.append(cat_1.loss.v.v)
#     ddloss_array.append(cat_1.loss.v_tt.v)
# loss_array = np.array(loss_array)
# ddloss_array = np.array(ddloss_array)

# src.model.t.v = -1.5
# plt.close()
# plt.plot(t_array, loss_array)
# plt.plot(t_array, ddloss_array)
# plt.xlabel("log(R)")
# plt.ylabel("loss")
# plt.axhline(0.0, ls='--')
