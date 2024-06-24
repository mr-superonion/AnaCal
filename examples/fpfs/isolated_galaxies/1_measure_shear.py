import anacal
import galsim
import numpy as np

nstamp = 100
seed = 2
pixel_scale = 0.2
noise_variance = 0.23

noise_array = None

rcut = 32
ngrid = rcut * 2
force_detect = False

if not force_detect:
    coords = None
    buff = 15
else:
    # force detection at center
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

fpfs_config = anacal.fpfs.FpfsConfig()


psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0)
psf_array = (
    psf_obj.shift(0.5 * pixel_scale, 0.5 * pixel_scale)
    .drawImage(nx=ngrid, ny=ngrid, scale=pixel_scale)
    .array
)
psf_array = psf_array[
    ngrid // 2 - rcut : ngrid // 2 + rcut,
    ngrid // 2 - rcut : ngrid // 2 + rcut,
]

out = []
for gname in ["g2-1", "g2-0"]:
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
        nrot_per_gal=1,
    )[0]

    out.append(
        anacal.fpfs.process_image(
            fpfs_config=fpfs_config,
            gal_array=gal_array,
            psf_array=psf_array,
            pixel_scale=pixel_scale,
            noise_variance=noise_variance,
            noise_array=noise_array,
            coords=coords,
        )
    )

e1_0 = out[0]["wdet"] * out[0]["e1"]
e1_1 = out[1]["wdet"] * out[1]["e1"]
e1g1_0 = out[0]["wdet_g1"] * out[0]["e1"] + out[0]["wdet"] * out[0]["e1_g1"],
e1g1_1 = out[1]["wdet_g1"] * out[1]["e1"] + out[1]["wdet"] * out[1]["e1_g1"],

print(
    (np.sum(e1_1) - np.sum(e1_0))
    / (np.sum(e1g1_1) + np.sum(e1g1_0))
)

e2_0 = out[0]["wdet"] * out[0]["e2"]
e2_1 = out[1]["wdet"] * out[1]["e2"]
e2g2_0 = out[0]["wdet_g2"] * out[0]["e2"] + out[0]["wdet"] * out[0]["e2_g2"],
e2g2_1 = out[1]["wdet_g2"] * out[1]["e2"] + out[1]["wdet"] * out[1]["e2_g2"],

print(
    (np.sum(e2_1) - np.sum(e2_0))
    / (np.sum(e2g2_1) + np.sum(e2g2_0))
)
