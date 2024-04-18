import anacal
import fpfs
import numpy as np
import pytest


def rfft2_to_fft2(rfft2_output, original_shape):
    """
    Transforms the output of np.fft.rfft2 (real input FFT) into the format
    of np.fft.fft2 (full complex FFT), given the original shape of the data.

    Parameters:
    rfft2_output: Output of np.fft.rfft2.
    original_shape: Shape of the original 2D data before FFT.

    Returns:
        A numpy array with the same shape and content that would have been
        obtained using np.fft.fft2 on the original real data.
    """
    # Create an empty array with the original shape for the full FFT result
    full_fft = np.zeros(original_shape, dtype=np.complex128)

    # The first half (positive frequencies) can be copied directly
    full_fft[:, :rfft2_output.shape[1]] = rfft2_output

    # For the negative frequencies (excluding DC and Nyquist if even-sized
    # signal)
    for x in range(1, original_shape[0]):
        for y in range(1, original_shape[1] - rfft2_output.shape[1]):
            full_fft[-x, -y] = np.conj(rfft2_output[x, y])

    # Handle the Nyquist frequency for even-sized signals
    if original_shape[1] % 2 == 0:
        full_fft[:, -1] = np.conj(full_fft[:, 1])

    return full_fft


@pytest.mark.parametrize("noise_std", [0.1, 0.2, 0.5])
def test_noise_sim_correlated(noise_std):
    ngrid = 32
    scale = 0.2
    d = np.pi * 0.8 / scale
    sigma = 1.0 / 3.0 / scale
    seed = 2
    gt_model = anacal.model.GaussianTopHat(d=d, sigma=sigma)
    theta = np.pi / 20
    gamma1 = 0.1
    gamma2 = -0.3
    gt_model.set_transform(theta=theta, gamma1=gamma1, gamma2=gamma2)
    corr_array = anacal.noise.simulate_noise_correlation(
        noise_std=noise_std,
        corr_model=gt_model,
        nx=ngrid,
        ny=ngrid,
        scale=scale,
    )

    np.testing.assert_almost_equal(
        corr_array[ngrid // 2, ngrid // 2],
        noise_std**2.0,
    )

    ngrid2 = 2048
    noise_array = anacal.noise.simulate_noise(
        seed=seed,
        correlation=corr_array,
        nx=ngrid2,
        ny=ngrid2,
        scale=scale,
    )
    np.testing.assert_allclose(np.std(noise_array), noise_std, rtol=1e-3)
    np.testing.assert_allclose(0.0, np.average(noise_array), atol=1e-3, rtol=0)
    noise_imag = rfft2_to_fft2(noise_array, (ngrid2, ngrid2)).imag
    np.testing.assert_almost_equal(noise_imag, 0.0)

    noise_array2 = anacal.noise.simulate_noise(
        seed=seed,
        correlation=corr_array,
        nx=ngrid2,
        ny=ngrid2,
        scale=scale,
        do_rotate=True,
    )
    np.testing.assert_allclose(np.std(noise_array2), noise_std, rtol=1e-3)
    np.testing.assert_allclose(0.0, np.average(noise_array2), atol=1e-3, rtol=0)

    np.testing.assert_almost_equal(
        fpfs.image.util.rotate90(noise_array)[1:, 1:],
        noise_array2[1:, 1:],
    )
    noise_imag = rfft2_to_fft2(noise_array2, (ngrid2, ngrid2)).imag
    np.testing.assert_almost_equal(noise_imag, 0.0)
    return


@pytest.mark.parametrize("noise_std", [0.1, 0.2, 0.5])
def test_noise_sim_white(noise_std):
    seed = 1
    ngrid2 = 1024
    noise_array = anacal.noise.simulate_noise(
        seed=seed,
        noise_std=noise_std,
        nx=ngrid2,
        ny=ngrid2,
    )
    np.testing.assert_allclose(np.std(noise_array), noise_std, rtol=1e-3)
    np.testing.assert_allclose(0.0, np.average(noise_array), atol=1e-3, rtol=0)
    return
