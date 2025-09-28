import anacal
import numpy as np


def smooth(
    sigmaf,
    kmax,
    img_array,
    psf_array,
    noise_array=None,
):
    """This function convolves an image to transform the PSF to a Gaussian

    Args:
    sigmaf (float):         Gaussian Sigma in Fourier space [1/pixel]
    kmax (float):           truncation in Fourier space [1/pixel]
    img_array (ndarray):    image data
    psf_array (ndarray):    psf data
    noise_array (ndarray):  noise data
    """

    ny, nx = img_array.shape
    # Fourier transform
    npady = (ny - psf_array.shape[0]) // 2
    npadx = (nx - psf_array.shape[1]) // 2

    if noise_array is not None:
        img_conv = np.fft.irfft2(
            (
                np.fft.rfft2(img_array)
                / np.fft.rfft2(
                    np.fft.ifftshift(
                        np.pad(
                            psf_array,
                            (npady, npadx),
                            mode="constant",
                        ),
                    )
                )
                + np.fft.rfft2(noise_array)
                / np.fft.rfft2(
                    np.fft.ifftshift(
                        np.pad(
                            anacal.utils.rotate90(psf_array),
                            (npady, npadx),
                            mode="constant",
                        ),
                    )
                )
            )
            * anacal.fpfs.gauss_kernel_rfft(
                ny,
                nx,
                sigmaf,
                kmax,
                return_grid=False,
            ),
            (ny, nx),
        )
    else:
        img_conv = np.fft.irfft2(
            np.fft.rfft2(img_array)
            / np.fft.rfft2(
                np.fft.ifftshift(
                    np.pad(
                        psf_array,
                        (npady, npadx),
                        mode="constant",
                    ),
                )
            )
            * anacal.fpfs.gauss_kernel_rfft(
                ny,
                nx,
                sigmaf,
                kmax,
                return_grid=False,
            ),
            (ny, nx),
        )
    return img_conv
