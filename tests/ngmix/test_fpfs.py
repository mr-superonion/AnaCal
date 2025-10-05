import anacal
import galsim
import numpy as np


def test_ngmix_fpfs():
    nx = 64
    ny = 64
    scale = 0.2
    psf_fwhm = 0.8
    sigma_shapelets = 0.38
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

    flux = 150.0
    angle = 30.0
    obj0 = galsim.Exponential(half_light_radius=0.21).shear(e1=0.2, e2=0)
    obj = obj0.rotate(angle * galsim.degrees).withFlux(flux)
    obj = obj.shift(0.5 * scale, 0.5 * scale)
    obj = galsim.Convolve(psf_obj, obj)

    # Create an empty image
    full_image = galsim.ImageF(ncol=nx, nrow=ny, scale=scale)

    center = (32, 32)
    shift = galsim.PositionD(
        (center[0] - nx / 2) * scale, (center[1] - ny / 2) * scale
    )
    final_galaxy = obj.shift(shift)
    final_galaxy.drawImage(image=full_image, add_to_image=True)
    img_array = full_image.array

    num_epochs = 20
    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_shapelets,
        stamp_size=32,
        fpfs_c0=0.0,
    )
    src = anacal.table.galNumber()
    src.model.x1.v = 32 * scale
    src.model.x2.v = 32 * scale
    src.model.F.v = 1.0
    src.model.t.v = -0.5
    catalog = [src]
    prior = anacal.ngmix.modelPrior()

    cat = fitter.process_block(
        catalog=catalog,
        img_array=img_array,
        psf_array=psf_array,
        prior=prior,
        num_epochs=num_epochs,
        variance=0.1,
    )[0]
    m00 = cat.fpfs_m0.v
    trace = cat.fpfs_m2.v
    m22c = cat.fpfs_e1.v * cat.fpfs_m0.v
    m22s = cat.fpfs_e2.v * cat.fpfs_m0.v

    sigma_shapelets_fpfs = sigma_shapelets * np.sqrt(2.0)
    bound = 35
    std = 0.2

    ftask = anacal.fpfs.FpfsTask(
        npix=64,
        pixel_scale=scale,
        sigma_shapelets=sigma_shapelets_fpfs,
        psf_array=psf_array,
        do_detection=True,
        noise_variance=std**2.0,
        bound=bound,
    )

    coords = np.array(
        [(ny / 2.0 , nx / 2.0)],
        dtype=[
            ("y", "f8"),
            ("x", "f8"),
        ],
    )

    src = ftask.run(
        gal_array=img_array,
        psf=psf_array,
        det=coords,
    )["data"]

    np.testing.assert_allclose(
        src["m00"][0], m00,
        atol=0.0, rtol=1e-4,
    )

    np.testing.assert_allclose(
        src["m00"][0] + src["m20"][0], trace,
        atol=0.0, rtol=1e-4,
    )

    np.testing.assert_allclose(
        src["m22c"][0], m22c,
        atol=0.0, rtol=1e-4,
    )

    np.testing.assert_allclose(
        src["m22s"][0], m22s,
        atol=0.0, rtol=1e-4,
    )
