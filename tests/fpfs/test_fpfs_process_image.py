import time

import anacal
import numpy as np

from ..fixtures import load


def test_fpfs_init():
    nstamp = 10  # nstamp x nstamp galaxies, as rendered in the fixture
    noise_seed = 1  # seed for noise
    pixel_scale = 0.2  # LSST image pixel scale
    # noise variance for r-bands 10 year LSST coadd (mag zero point at 30)
    noise_std = 0.37
    noise_variance = noise_std**2.0

    rcut = 32  # cutout radius

    # Simulation
    ngrid = rcut * 2
    buff = 15

    fpfs_config = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=0.52,  # The first measurement scale
        sigma_shapelets2=0.45,  # The second measurement scale
    )

    # Pre-rendered by tests/data/make_fixtures.py: a Moffat PSF (beta
    # 3.5, fwhm 0.6) and an nstamp x nstamp grid of COSMOS galaxies
    # sheared by g1 = +0.02 (seed 2, 4 rotations per galaxy, mag_zero
    # 30), padded by `buff` pixels on every side.
    fix = load("fpfs_process_image")
    psf_array = fix["psf"]
    gal_array = fix["gal"]
    assert gal_array.shape == (ngrid * nstamp + 2 * buff,) * 2

    # Add noise to galaxy image
    gal_array = gal_array + np.random.RandomState(noise_seed).normal(
        scale=noise_std,
        size=gal_array.shape,
    )
    # The pure noise for noise bias correction
    # make sure that the random seeds are different
    # (noise variance are the same)
    additional_noise_seed = int(noise_seed + 1e6)
    noise_array = np.random.RandomState(additional_noise_seed).normal(
        scale=noise_std,
        size=gal_array.shape,
    )
    # Detection is external now (the AnaCal detector owns detection); the
    # simulation puts one galaxy at the centre of each ngrid x ngrid stamp,
    # so the detection catalogue is simply the stamp-centre grid.
    centers = np.arange(nstamp) * ngrid + ngrid // 2 + buff
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    detection = np.zeros(
        nstamp * nstamp, dtype=[("y", np.float64), ("x", np.float64)]
    )
    detection["y"] = yy.ravel()
    detection["x"] = xx.ravel()
    out1 = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=30.0,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        noise_variance=max(noise_variance, 0.23),
        noise_array=noise_array,
        detection=detection,
    )
    # Native per-source PSF path: a 1x1 stamp grid holds the same PSF
    # everywhere, so the C++ ForceTask drawing per source must equal the
    # constant-PSF path bit for bit.
    grid_model = anacal.psf.GridPsfModel(
        stamps=np.ascontiguousarray(
            psf_array[None, None, :, :], dtype=np.float64
        ),
        dx=float(gal_array.shape[1]),
        dy=float(gal_array.shape[0]),
    )
    t0 = time.time()
    out1 = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=30.0,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        noise_variance=max(noise_variance, 0.23),
        noise_array=noise_array,
        detection=detection,
    )
    t1 = time.time()
    print("constant-PSF time: ", t1 - t0)
    out2 = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=30.0,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        noise_variance=max(noise_variance, 0.23),
        noise_array=noise_array,
        detection=detection,
        psf_model=grid_model,
    )
    t2 = time.time()
    print("native per-source time: ", t2 - t1)
    assert np.all(out1 == out2)
    return


if __name__ == "__main__":
    test_fpfs_init()
