import anacal
import galsim
import numpy as np

nstamp = 100
seed = 1
pixel_scale = 0.2
noise_variance = 0.23

fpfs_config = anacal.fpfs.FpfsConfig()
noise_array = None
cov_matrix = None
coords = None

rcut = fpfs_config.rcut
ngrid = rcut * 2

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
for gname in ["g1-1", "g1-0"]:
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
        buff=15,
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
