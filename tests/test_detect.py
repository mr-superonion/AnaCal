import anacal
import galsim
import numpy as np

kwargs = {
    "sigma_arcsec": 0.5,
    "f_min": 0.14,
    "omega_f": 0.1,
    "v_min": 0.015,
    "omega_v": 0.020,
    "pthres": 0.012,
}


def test_detect(mag=27.5):
    nx = 64
    ny = 64
    scale = 0.2
    psf_fwhm = 0.7
    sigma_arcsec = 0.5
    klim = 100.0
    flux = 10 ** ((30 - mag) / 2.5)

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

    obj0 = galsim.Exponential(half_light_radius=0.25)
    obj0 = obj0.shear(e1=0.1, e2=-0.15).withFlux(flux)

    block = anacal.geometry.get_block_list(
        nx,
        ny,
        nx,
        ny,
        0,
        scale,
    )[0]

    def make_sim(g1, g2, angle=0.0):
        obj = obj0.rotate(angle * galsim.degrees).shear(g1=g1, g2=g2)
        obj = obj.shift(0.5 * scale, 0.5 * scale)
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

    cats = anacal.detector.find_peaks(
        img_array=make_sim(g1=-0.02, g2=0.0),
        psf_array=psf_array,
        block=block,
        noise_array=None,
        image_bound=0,
        **kwargs,
    )
    assert len(cats) == 1
    assert cats[0].model.x1.v / scale == nx // 2
    assert cats[0].model.x2.v / scale == ny // 2

    cat_1 = anacal.detector.find_peaks(
        img_array=make_sim(g1=-0.02, g2=0.0),
        psf_array=psf_array,
        block=block,
        noise_array=None,
        image_bound=0,
        **kwargs,
    )[0]

    cat_2 = anacal.detector.find_peaks(
        img_array=make_sim(g1=0.02, g2=0.0),
        psf_array=psf_array,
        block=block,
        noise_array=None,
        image_bound=0,
        **kwargs,
    )[0]

    np.testing.assert_approx_equal(
        (cat_2.wdet.v - cat_1.wdet.v) / 0.04,
        (cat_2.wdet.g1 + cat_1.wdet.g1) / 2.0,
        3,
    )

    cat_1 = anacal.detector.find_peaks(
        img_array=make_sim(g2=-0.02, g1=0.0),
        psf_array=psf_array,
        block=block,
        noise_array=None,
        image_bound=0,
        **kwargs,
    )[0]

    cat_2 = anacal.detector.find_peaks(
        img_array=make_sim(g2=0.02, g1=0.0),
        psf_array=psf_array,
        block=block,
        noise_array=None,
        image_bound=0,
        **kwargs,
    )[0]

    np.testing.assert_approx_equal(
        (cat_2.wdet.v - cat_1.wdet.v) / 0.04,
        (cat_2.wdet.g2 + cat_1.wdet.g2) / 2.0,
        3,
    )

    for ang in np.random.random(10) * 360:
        cat_1 = anacal.detector.find_peaks(
            img_array=make_sim(g1=-0.00, g2=0.0, angle=ang),
            psf_array=psf_array,
            block=block,
            noise_array=None,
            image_bound=0,
            **kwargs,
        )[0]

        cat_2 = anacal.detector.find_peaks(
            img_array=make_sim(g1=-0.00, g2=0.0, angle=ang + 90.0),
            psf_array=psf_array,
            block=block,
            noise_array=None,
            image_bound=0,
            **kwargs,
        )[0]

        np.testing.assert_approx_equal(
            cat_1.wdet.v,
            cat_2.wdet.v,
            6,
        )
    return
