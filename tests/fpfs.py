import fpfs
import jax.numpy as jnp


def smooth(
    task,
    img_array,
    psf_array,
    noise_array=None,
):
    """This function convolves an image to transform the PSF to a Gaussian

    Args:
    img_array (ndarray):     image data
    psf_array (ndarray):     psf data
    noise_array (ndarray):   noise data
    """

    ny, nx = img_array.shape
    # Fourier transform
    npady = (ny - psf_array.shape[0]) // 2
    npadx = (nx - psf_array.shape[1]) // 2

    if noise_array is not None:
        img_conv = jnp.fft.irfft2(
            (
                jnp.fft.rfft2(img_array)
                / jnp.fft.rfft2(
                    jnp.fft.ifftshift(
                        jnp.pad(
                            psf_array,
                            (npady, npadx),
                            mode="constant",
                        ),
                    )
                )
                + jnp.fft.rfft2(noise_array)
                / jnp.fft.rfft2(
                    jnp.fft.ifftshift(
                        jnp.pad(
                            fpfs.image.util.rotate90(psf_array),
                            (npady, npadx),
                            mode="constant",
                        ),
                    )
                )
            )
            * fpfs.image.util.gauss_kernel_rfft(
                ny,
                nx,
                task.sigmaf,
                task.klim,
                return_grid=False,
            ),
            (ny, nx),
        )
    else:
        img_conv = jnp.fft.irfft2(
            jnp.fft.rfft2(img_array)
            / jnp.fft.rfft2(
                jnp.fft.ifftshift(
                    jnp.pad(
                        psf_array,
                        (npady, npadx),
                        mode="constant",
                    ),
                )
            )
            * fpfs.image.util.gauss_kernel_rfft(
                ny,
                nx,
                task.sigmaf,
                task.klim,
                return_grid=False,
            ),
            (ny, nx),
        )
    return img_conv
