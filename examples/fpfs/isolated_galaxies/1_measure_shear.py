import anacal
import galsim
import numpy as np

nstamp = 20  # nstamp x nstamp galaxies
seed = 2     # seed for galaxy
noise_seed = 1  # seed for noise
pixel_scale = 0.2  # LSST image pixel scale
# noise variance for r-bands 10 year LSST coadd (magnitude zero point at 30)
noise_variance = 0.37
# NOTE: We can set noise variance to zero in the image simulation, but
# we cannot set that to zero in the measurement. The measurement needs a
# non-zero image noise variance to be run
# For the test with noiseless image simulation, we can set do_add_noise
# to False, but keep the noise variance to the realistic one as the input
# of the measurement
do_add_noise = False  # Add image noise or not
do_force_detect = True  # Force to have a detection at the center

rcut = 32  # cutout radius
test_component = 1  # which shear component to test
nrot_per_gal = 4  # number of rotation for each galaxy

# Simulation
ngrid = rcut * 2
if not do_force_detect:
    coords = None
    buff = 15
else:
    #
    # Force to have a detection at the center
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
            ("is_peak", np.int32),
            ("mask_value", np.int32),
        ]
    )
    coords = np.empty(ns, dtype=dtype)
    coords["y"] = yx[0]
    coords["x"] = yx[1]
    coords["is_peak"] = np.ones(ns)
    coords["mask_value"] = np.zeros(ns)

fpfs_config = anacal.fpfs.FpfsConfig(
    sigma_arcsec=0.52,      # The first measurement scale (also for detection)
    sigma_arcsec2=0.45,     # The second measurement scale
)


psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0)
psf_array = (
    psf_obj.shift(0.5 * pixel_scale, 0.5 * pixel_scale)
    .drawImage(nx=ngrid, ny=ngrid, scale=pixel_scale)
    .array
)

# Measurement
out = []
for gname in ["g%d-1" % test_component, "g%d-0" % test_component]:
    gal_array = anacal.simulation.make_isolated_sim(
        gal_type="mixed",
        sim_method="fft",
        psf_obj=psf_obj,
        gname=gname,
        seed=seed,
        ny=ngrid * nstamp,
        nx=ngrid * nstamp,
        scale=pixel_scale,
        do_shift=False,
        buff=buff,
        nrot_per_gal=nrot_per_gal,
        mag_zero=30,
    )[0]

    if do_add_noise:
        noise_std = np.sqrt(noise_variance)
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
            gal_array=gal_array,
            psf_array=psf_array,
            pixel_scale=pixel_scale,
            noise_variance=max(noise_variance, 0.23),
            noise_array=noise_array,
            coords=coords,
        )
    )

# Printing the results
print("Testing for shear component: %d" % test_component)
print("Measurement with sigma_arcsec=%.2f:" % fpfs_config.sigma_arcsec)
ename = "e%d" % test_component
egname = "e%d_g%d" % (test_component, test_component)
wgname = "w_g%d" % test_component
e1_0 = out[0]["w"] * out[0][ename]
e1_1 = out[1]["w"] * out[1][ename]
e1g1_0 = out[0][wgname] * out[0][ename] + out[0]["w"] * out[0][egname],
e1g1_1 = out[1][wgname] * out[1][ename] + out[1]["w"] * out[1][egname],

mbias = (np.sum(e1_0) - np.sum(e1_1)) / (np.sum(e1g1_0) + np.sum(e1g1_1)) \
    / 0.02 - 1  # 0.02 is the input shear
print(
    "    Multiplicative bias is %.3f e-3" % (mbias * 1e3)
)
cbias = (np.sum(e1_0) + np.sum(e1_1)) / (np.sum(e1g1_0) + np.sum(e1g1_1))
print(
    "    Additive bias is %.3f e-5" % (cbias * 1e5)
)
assert mbias < 2e-3

print("Measurement with sigma_arcsec=%.2f:" % fpfs_config.sigma_arcsec2)
ename = "e%d_2" % test_component
egname = "e%d_g%d_2" % (test_component, test_component)
e1_0 = out[0]["w"] * out[0][ename]
e1_1 = out[1]["w"] * out[1][ename]
e1g1_0 = out[0][wgname] * out[0][ename] + out[0]["w"] * out[0][egname],
e1g1_1 = out[1][wgname] * out[1][ename] + out[1]["w"] * out[1][egname],

mbias = (np.sum(e1_0) - np.sum(e1_1)) / (np.sum(e1g1_0) + np.sum(e1g1_1)) \
    / 0.02 - 1  # 0.02 is the input shear
print(
    "    Multiplicative bias is %.3f e-3" % (mbias * 1e3)
)
cbias = (np.sum(e1_0) + np.sum(e1_1)) / (np.sum(e1g1_0) + np.sum(e1g1_1))
print(
    "    Additive bias is %.3f e-5" % (cbias * 1e5)
)
assert mbias < 2e-3
