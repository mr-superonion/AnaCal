import anacal
import galsim
import numpy as np

nstamp = 10
seed = 1
pixel_scale = 0.2
noise_variance = 0.23

noise_array = None
cov_matrix = None

rcut = 32
ngrid = rcut * 2
force_detect = True

if force_detect:
    coords = None
    buff = 15
else:
    # force detection at center
    indx = np.arange(ngrid // 2, ngrid * nstamp, ngrid)
    indy = np.arange(ngrid // 2, ngrid * nstamp, ngrid)
    inds = np.meshgrid(indy, indx, indexing="ij")
    coords = np.vstack(inds).T
    buff = 0
fpfs_config = anacal.fpfs.FpfsConfig(
    force=force_detect, rcut=rcut,
    gmeasure=3,
)


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

outcomes = []
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
        simple_sim=False,
    )[0]

    outcomes.append(
        anacal.fpfs.process_image(
            fpfs_config,
            gal_array,
            psf_array,
            pixel_scale,
            noise_variance,
            noise_array,
            cov_matrix,
            coords,
        )
    )


print(
    (np.sum(outcomes[1][:, 0]) - np.sum(outcomes[0][:, 0]))
    / (np.sum(outcomes[1][:, 1]) + np.sum(outcomes[0][:, 1]))
)
print(
    (np.sum(outcomes[1][:, 2]) - np.sum(outcomes[0][:, 2]))
    / (np.sum(outcomes[1][:, 3]) + np.sum(outcomes[0][:, 3]))
)
