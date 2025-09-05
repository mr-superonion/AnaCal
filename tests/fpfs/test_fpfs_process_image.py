import time

import anacal
import galsim
import numpy as np


class MyPsf(anacal.psf.BasePsf):
    def __init__(self, psf_array):
        super().__init__()
        self.psf_array = psf_array

    def draw(self, x, y):
        return self.psf_array


def test_fpfs_init():
    nstamp = 30  # nstamp x nstamp galaxies
    seed = 2  # seed for galaxy
    noise_seed = 1  # seed for noise
    pixel_scale = 0.2  # LSST image pixel scale
    # noise variance for r-bands 10 year LSST coadd (mag zero point at 30)
    noise_std = 0.37
    noise_variance = noise_std**2.0

    rcut = 32  # cutout radius
    test_component = 1  # which shear component to test
    nrot_per_gal = 4  # number of rotation for each galaxy

    # Simulation
    ngrid = rcut * 2
    buff = 15

    fpfs_config = anacal.fpfs.FpfsConfig(
        sigma_arcsec=0.52,  # The first measurement scale (also for detection)
        sigma_arcsec2=0.45,  # The second measurement scale
    )

    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0)
    psf_array = (
        psf_obj.shift(0.5 * pixel_scale, 0.5 * pixel_scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=pixel_scale)
        .array
    )

    gname = "g%d-1" % test_component
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
    out1 = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=30.0,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        noise_variance=max(noise_variance, 0.23),
        noise_array=noise_array,
    )
    psf_object = MyPsf(psf_array=psf_array)
    t0 = time.time()
    out1 = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=30.0,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        noise_variance=max(noise_variance, 0.23),
        noise_array=noise_array,
    )
    t1 = time.time()
    print("C++ Time: ", t1 - t0)
    out2 = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=30.0,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        noise_variance=max(noise_variance, 0.23),
        noise_array=noise_array,
        psf_object=psf_object,
    )
    t2 = time.time()
    print("Python Time: ", t2 - t1)
    assert np.all(out1 == out2)
    return


if __name__ == "__main__":
    test_fpfs_init()
