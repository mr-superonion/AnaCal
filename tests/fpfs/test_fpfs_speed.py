import anacal
import galsim
import numpy as np
from memory_profiler import memory_usage

from .. import mem_used, print_mem

ny = 5000
nx = 5000

std = 0.2
bound = 40
mag_zero = 30.0
pixel_scale = 0.2

sigma_shapelets = 0.55
psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
    e1=0.02, e2=-0.02
)
psf_array = (
    psf_obj.shift(0.5 * pixel_scale, 0.5 * pixel_scale)
    .drawImage(nx=64, ny=64, scale=pixel_scale)
    .array
).astype(np.float64)
gal_obj = (
    psf_obj.shift(-3.5, 2) * 2
    + psf_obj.shift(2, -1) * 4
    + psf_obj.shift(-2, -0.5) * 4
    + psf_obj.shift(-3.2, 0.5) * 6
)
gal_array = gal_obj.drawImage(
    nx=nx,
    ny=ny,
    scale=pixel_scale,
).array.astype(np.float64)


def func():
    noise_array = np.random.randn(ny, nx)
    # t0 = time.time()
    noise_variance = std**2.0

    ftask = anacal.fpfs.FpfsTask(
        npix=64,
        pixel_scale=pixel_scale,
        sigma_shapelets=sigma_shapelets,
        psf_array=psf_array,
        do_detection=True,
        noise_variance=noise_variance,
        bound=bound,
    )

    # t1 = time.time()
    # print("Detection Time: ", t1 - t0)
    det = ftask.detect(
        gal_array=gal_array,
        fthres=5.0,
        pthres=anacal.fpfs.fpfs_det_sigma2 + 0.02,
        omega_v=0.2557259133327884 * 1.6,
        v_min=0.2557259133327884 * 0.8,
        noise_array=noise_array,
    )[0:10000]
    del det
    return


def test_speed():
    print("")
    initial_memory_usage = mem_used()

    print("Initial Mem:")
    print_mem(initial_memory_usage)
    func()

    peak_memory_usage = max(memory_usage(proc=(func,)))
    print("Peak Mem:", peak_memory_usage, "M")
    final_memory_usage = mem_used()
    print("Additional Mem:")
    print_mem(final_memory_usage - initial_memory_usage)
    return


if __name__ == "__main__":
    test_speed()
