import anacal
import numpy as np

from ..fixtures import load


def test_noise_variance():
    stamp_size = 600
    scale = 0.2

    # pre-rendered sheared Moffat PSF (beta 2.5, fwhm 0.6), 64 x 64
    psf_array = load("image_noise_variance")["psf"]
    noise_variance = 0.2 ** 2.0
    img_array = np.random.RandomState(0).normal(
        scale=np.sqrt(noise_variance),
        size=(stamp_size, stamp_size),
    )
    sigma_arcsec = 0.52
    img_obj = anacal.image.ImageQ(
        nx=stamp_size,
        ny=stamp_size,
        scale=scale,
        sigma_arcsec=sigma_arcsec,
        klim=3.0 / 0.2,
    )
    img_smoothed = img_obj.prepare_qnumber_image(
        img_array,
        psf_array,
        xcen=stamp_size//2,
        ycen=stamp_size//2,
    )[0]
    std1 = np.std(img_smoothed)

    std2 = np.sqrt(
        anacal.image.get_smoothed_variance(
            scale,
            sigma_arcsec,
            psf_array,
            noise_variance,
        )
    )
    np.testing.assert_allclose(std1, std2, atol=1e-3, rtol=1e-3)
    return
