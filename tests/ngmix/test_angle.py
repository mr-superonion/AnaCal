import anacal
import galsim
import numpy as np


def test_angle():
    nx = 64
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
        psf_obj.shift(0.5 * scale, 0.5 * scale).drawImage(
            nx=nx,
            ny=ny,
            scale=scale,
        )
        .array
    )

    flux = 150.0
    angle = 30.0
    obj0 = galsim.Exponential(half_light_radius=0.16).shear(e1=0.2, e2=0)
    obj = obj0.rotate(angle * galsim.degrees).withFlux(flux)
    obj = obj.shift(0.5 * scale, 0.5 * scale)
    obj = galsim.Convolve(psf_obj, obj)

    # Create an empty image
    full_image = galsim.ImageF(ncol=nx, nrow=ny, scale=scale)

    center = (ny // 2, nx // 2)
    shift = galsim.PositionD(
        (center[0] - nx // 2) * scale, (center[1] - ny // 2) * scale
    )
    img_array = obj.shift(shift).drawImage(
        image=full_image, add_to_image=True
    ).array

    num_epochs = 20
    fitter = anacal.ngmix.GaussFit(
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        stamp_size=32,
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

    np.testing.assert_allclose(
        cat.model.t.v / np.pi * 180, angle,
        atol=0.0, rtol=1e-4,
    )
