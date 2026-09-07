import os

import anacal
import fitsio
import numpy as np

nstamp = 10  # nstamp x nstamp galaxies, as stored in isolated_sim.fits
noise_seed = 1  # seed for noise
pixel_scale = 0.2  # LSST image pixel scale
# noise variance for r-bands 10 year LSST coadd (magnitude zero point at 30)
noise_std = 0.37
noise_variance = noise_std**2.0
# NOTE: We can set noise variance to zero in the image simulation, but
# we cannot set that to zero in the measurement. The measurement needs a
# non-zero image noise variance to be run
# For the test with noiseless image simulation, we can set do_add_noise
# to False, but keep the noise variance to the realistic one as the input
# of the measurement
do_add_noise = False  # Add image noise or not

rcut = 32  # cutout radius
test_component = 1  # which shear component to test

# Simulation
ngrid = rcut * 2
#
# FPFS no longer detects internally: detection is done with the AnaCal
# detector (anacal.task.Task / detector.h) on real data.  Here the
# galaxies sit at the stamp centres, so the detection catalogue is
# simply the stamp-centre grid.
#
indx = np.arange(ngrid // 2, ngrid * nstamp, ngrid)
indy = np.arange(ngrid // 2, ngrid * nstamp, ngrid)
ns = len(indx) * len(indy)
inds = np.meshgrid(indy, indx, indexing="ij")
yx = np.vstack([np.ravel(_) for _ in inds])
buff = 0
dtype = np.dtype(
    [
        ("y", np.int32),
        ("x", np.int32),
    ]
)
detection = np.empty(ns, dtype=dtype)
detection["y"] = yx[0]
detection["x"] = yx[1]

fpfs_config = anacal.fpfs.FpfsConfig(
    sigma_shapelets1=0.52,  # kernel 1 (required)
    sigma_shapelets2=0.55,  # kernel 2 (optional)
)


# Pre-rendered with GalSim by tests/data/make_fixtures.py: a Moffat PSF
# (beta 3.5, fwhm 0.6) and, for each shear sign, an nstamp x nstamp grid
# of COSMOS galaxies (4 rotations per galaxy, seed 2) at 0.2 arcsec/pixel.
data_dir = os.path.dirname(os.path.abspath(__file__))
sim_file = os.path.join(data_dir, "isolated_sim.fits")
psf_array = fitsio.read(sim_file, ext="psf")

# Measurement
out = []
for gname in ["g%d-1" % test_component, "g%d-0" % test_component]:
    gal_array = fitsio.read(sim_file, ext="gal_%s" % gname)

    if do_add_noise:
        # Add noise to galaxy image
        gal_array = gal_array + np.random.RandomState(noise_seed).normal(
            scale=noise_std,
            size=gal_array.shape,
        )
        # The pure noise for noise bias correction
        # make sure that the random seeds are different
        # (noise variance are the same)
        add_noise_seed = int(noise_seed + 1e6)
        noise_array = np.random.RandomState(add_noise_seed).normal(
            scale=noise_std,
            size=gal_array.shape,
        )
    else:
        noise_array = None
    out.append(
        anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            mag_zero=30.0,
            gal_array=gal_array,
            psf_array=psf_array,
            pixel_scale=pixel_scale,
            noise_variance=max(noise_variance, 0.23),
            noise_array=noise_array,
            detection=detection,
        )
    )

# Printing the results.  Every galaxy is force-measured at its stamp
# centre, so no selection weight is needed (the detection/selection
# weight lives in the AnaCal detector path on real data).
print("Testing for shear component: %d" % test_component)

for prefix, sigma in [
    ("fpfs1", fpfs_config.sigma_shapelets1),
    ("fpfs2", fpfs_config.sigma_shapelets2),
]:
    print("Measurement with sigma_shapelets=%.2f:" % sigma)
    ename = "%s_e%d" % (prefix, test_component)
    egname = "%s_de%d_dg%d" % (prefix, test_component, test_component)
    e1_0 = out[0][ename]
    e1_1 = out[1][ename]
    e1g1_0 = out[0][egname]
    e1g1_1 = out[1][egname]

    mbias = (np.sum(e1_0) - np.sum(e1_1)) / (
        np.sum(e1g1_0) + np.sum(e1g1_1)
    ) / 0.02 - 1  # 0.02 is the input shear
    print("    Multiplicative bias is %.3f e-3" % (mbias * 1e3))
    cbias = (np.sum(e1_0) + np.sum(e1_1)) / (np.sum(e1g1_0) + np.sum(e1g1_1))
    print("    Additive bias is %.3f e-5" % (cbias * 1e5))
    assert mbias < 2e-3
