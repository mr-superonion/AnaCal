"""anacal.fpfs.process_image on the noisy COSMOS scene: the native
per-source PSF path (a GridPsfModel holding one stamp) must equal the
constant-PSF path bit for bit, and the 2nd-order estimator
(shape_order=2, 9 modes) must reproduce the 4th-order run's shared
columns bit for bit with the q family NaN.
"""

import time

import anacal
import numpy as np

from ..fixtures import cosmos_centres, cosmos_scene


def test_grid_psf_matches_constant_psf():
    noise_std = 0.37
    noise_seed = 1
    pixel_scale = 0.2
    fpfs_config = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=0.52,  # The first measurement scale
        sigma_shapelets2=0.45,  # The second measurement scale
        shape_order=4,
    )
    gal_array, psf_array = cosmos_scene()
    gal_array = gal_array + np.random.RandomState(noise_seed).normal(
        scale=noise_std, size=gal_array.shape
    )
    # the pure noise for the noise bias correction, from a different seed
    noise_array = np.random.RandomState(noise_seed + 10**6).normal(
        scale=noise_std, size=gal_array.shape
    )
    detection = cosmos_centres()
    kwargs = dict(
        fpfs_config=fpfs_config,
        mag_zero=30.0,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        noise_variance=max(noise_std**2, 0.23),
        noise_array=noise_array,
        detection=detection,
    )
    t0 = time.time()
    out1 = anacal.fpfs.process_image(**kwargs)
    t1 = time.time()
    print("constant-PSF time: ", t1 - t0)
    # a 1 x 1 stamp grid holds the same PSF everywhere, so the C++
    # ForceTask drawing per source must equal the constant-PSF path
    grid_model = anacal.psf.GridPsfModel(
        stamps=np.ascontiguousarray(
            psf_array[None, None, :, :], dtype=np.float64
        ),
        dx=float(gal_array.shape[1]),
        dy=float(gal_array.shape[0]),
    )
    out2 = anacal.fpfs.process_image(psf_model=grid_model, **kwargs)
    t2 = time.time()
    print("native per-source time: ", t2 - t1)
    assert len(out1) == len(detection)
    assert np.all(out1 == out2)

    # shape_order=2 (the default): the 9 modes through m44 are the same
    # sums as in the 12-mode run, so every column outside the 4th-order
    # family is bit-identical; q and its responses are NaN
    kwargs["fpfs_config"] = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=0.52, sigma_shapelets2=0.45
    )
    assert kwargs["fpfs_config"].shape_order == 2
    t2 = time.time()
    out3 = anacal.fpfs.process_image(**kwargs)
    print("shape_order=2 time: ", time.time() - t2)
    assert out3.dtype == out1.dtype
    nan_tail = ("q1", "q2", "dq1_dg1", "dq1_dg2", "dq2_dg1", "dq2_dg2",
                "dm42c_dg1", "dm42c_dg2", "dm42s_dg1", "dm42s_dg2")
    for name in out1.dtype.names:
        if name.split("_", 1)[1] in nan_tail:
            assert np.all(np.isnan(out3[name])), name
            assert np.all(np.isfinite(out1[name])), name
        else:
            assert np.all(out3[name] == out1[name]), name


def test_grid_psf_matches_constant_psf_spectral():
    """Spectral renoising: the per-source-PSF path must also equal the
    constant-PSF path bit for bit, because the renoise kernels (rho,
    eta_0, tau_j) are built once per call from the per-cell psf_array;
    only the deconvolution filter chi / p is per source."""
    noise_std = 0.37
    noise_seed = 1
    pixel_scale = 0.2
    fpfs_config = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=0.52,
        sigma_shapelets2=0.45,
    )
    gal_array, psf_array = cosmos_scene()
    rng = np.random.RandomState(noise_seed)
    gal_array = gal_array + rng.normal(scale=noise_std, size=gal_array.shape)
    noise_array = np.random.RandomState(noise_seed + 10**6).normal(
        scale=noise_std, size=gal_array.shape
    )
    noise2_array = np.random.RandomState(noise_seed + 2 * 10**6).normal(
        scale=noise_std, size=gal_array.shape
    )
    detection = cosmos_centres()
    kwargs = dict(
        fpfs_config=fpfs_config,
        mag_zero=30.0,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        noise_variance=max(noise_std**2, 0.23),
        noise_array=noise_array,
        noise2_array=noise2_array,
        detection=detection,
        renoise_mode="spectral",
    )
    out1 = anacal.fpfs.process_image(**kwargs)
    grid_model = anacal.psf.GridPsfModel(
        stamps=np.ascontiguousarray(
            psf_array[None, None, :, :], dtype=np.float64
        ),
        dx=float(gal_array.shape[1]),
        dy=float(gal_array.shape[0]),
    )
    out2 = anacal.fpfs.process_image(psf_model=grid_model, **kwargs)
    assert len(out1) == len(detection)
    # default shape_order=2: the q family is NaN in both, so compare per
    # column (assert_array_equal is exact and treats NaN as equal)
    for name in out1.dtype.names:
        np.testing.assert_array_equal(out1[name], out2[name], err_msg=name)
