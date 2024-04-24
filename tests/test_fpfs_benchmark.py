import gc
import time

import anacal
import galsim
import numpy as np
from memory_profiler import memory_usage

from . import mem_used, print_mem

ny = 5000
nx = 5000

pthres = 0.2
pratio = 0.05
std = 0.4
det_nrot = 4
bound = 40

scale = 0.2
sigma_as = 0.55
psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
    e1=0.02, e2=-0.02
)
psf_data = (
    psf_obj.shift(0.5 * scale, 0.5 * scale)
    .drawImage(nx=64, ny=64, scale=scale)
    .array
).astype(np.float64)
gal_obj = (
    psf_obj.shift(-3.5, 2) * 2
    + psf_obj.shift(2, -1) * 4
    + psf_obj.shift(-2, -0.5) * 4
    + psf_obj.shift(-3.2, 0.5) * 6
)
gal_data = gal_obj.drawImage(nx=nx, ny=ny, scale=scale).array.astype(np.float64)


def test_detect():
    print("")
    initial_memory_usage = mem_used()

    def func():
        noise_data = np.random.randn(ny, nx)
        t0 = time.time()

        cov_element = np.ones((21, 21)) * std**2.0
        dtask = anacal.fpfs.FpfsDetect(
            nx=nx,
            ny=ny,
            psf_array=psf_data,
            pixel_scale=scale,
            cov_matrix=cov_element,
            sigma_arcsec=sigma_as,
            det_nrot=det_nrot,
        )
        t1 = time.time()
        print("Detection Time: ", t1 - t0)
        det = dtask.run(
            gal_array=gal_data,
            fthres=1.0,
            pthres=pthres,
            pratio=pratio,
            pthres2=anacal.fpfs.fpfs_det_sigma2 + 0.02,
            bound=bound,
            noise_array=noise_data,
        )[0:30000]
        mtask = anacal.fpfs.FpfsMeasure(
            psf_array=psf_data,
            pixel_scale=scale,
            sigma_arcsec=sigma_as,
            det_nrot=det_nrot,
        )
        src = mtask.run(
            gal_array=gal_data,
            det=det,
        )
        t2 = time.time()
        print("Final Time: ", t2 - t0)
        del noise_data, det, src, dtask, mtask
        return

    print_mem(initial_memory_usage)
    func()

    peak_memory_usage = max(memory_usage(proc=(func,)))
    print("Peak Mem:", peak_memory_usage, "M")
    gc.collect()
    final_memory_usage = mem_used()
    print_mem(final_memory_usage - initial_memory_usage)
    return


if __name__ == "__main__":
    test_detect()
