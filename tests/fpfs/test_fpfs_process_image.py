"""anacal.fpfs.process_image on the noisy COSMOS scene: the native
per-source PSF path (a GridPsfModel holding one stamp) must equal the
constant-PSF path bit for bit.
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
