import fpfs
import anacal
import numpy as np


def test_noise_sim():
    ngrid = 32
    scale = 0.2
    d = np.pi * 0.8 / scale
    sigma = 1.0 / 3.0 / scale
    noise_std = 0.5
    seed = 1
    gt_model = anacal.model.GaussianTopHat(d=d, sigma=sigma)
    corr_array = anacal.noise.simulate_noise_correlation(
        noise_std=noise_std,
        corr_model=gt_model,
        nx=ngrid, ny=ngrid,
        scale=scale,
    )

    np.testing.assert_almost_equal(
        corr_array[ngrid//2, ngrid//2],
        noise_std ** 2.0,
    )

    ngrid2 = 1024
    noise_array = anacal.noise.simulate_noise(
        seed=seed,
        correlation=corr_array,
        nx=ngrid2, ny=ngrid2, scale=0.2,
    )
    np.testing.assert_allclose(np.std(noise_array), noise_std, rtol=1e-3)
    np.testing.assert_allclose(0.0, np.average(noise_array), atol=1e-3, rtol=0)

    noise_array2 = anacal.noise.simulate_noise(
        seed=seed,
        correlation=corr_array,
        nx=ngrid2, ny=ngrid2, scale=0.2,
        do_rotate=True,
    )
    np.testing.assert_allclose(np.std(noise_array2), noise_std, rtol=1e-3)
    np.testing.assert_allclose(0.0, np.average(noise_array2), atol=1e-3, rtol=0)

    np.testing.assert_almost_equal(
        fpfs.image.util.rotate90(noise_array)[1:, 1:],
        noise_array2[1:, 1:],
    )
    return


