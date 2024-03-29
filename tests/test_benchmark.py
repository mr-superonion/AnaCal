import gc
import os
import time

import anacal
import fpfs
import galsim
import numpy as np
from memory_profiler import memory_usage

from . import mem_used, print_mem

ny = 5000
nx = 5000

pthres = 0.2
pratio = 0.05
std = 0.4

scale = 0.2
sigma_as = 0.55
psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(e1=0.02, e2=-0.02)
psf_data = (
    psf_obj.shift(0.5 * scale, 0.5 * scale).drawImage(nx=64, ny=64, scale=scale).array
)
gal_obj = (
    psf_obj.shift(-3.5, 2) * 2
    + psf_obj.shift(2, -1) * 4
    + psf_obj.shift(-2, -0.5) * 4
    + psf_obj.shift(-3.2, 0.5) * 6
)
gal_data = gal_obj.drawImage(nx=nx, ny=ny, scale=scale).array

tt = fpfs.image.measure_source(
    psf_data,
    pix_scale=scale,
    sigma_arcsec=sigma_as,
    nord=4,
    det_nrot=4,
)


def test_detect():
    initial_memory_usage = mem_used()
    task = anacal.fpfs.FpfsImage(
        nx=nx,
        ny=ny,
        scale=scale,
        sigma_arcsec=sigma_as,
        klim=10.0,
        psf_array=psf_data,
    )
    print("")

    def func():
        noise_data = np.random.randn(ny, nx)
        smooth_data = task.smooth_image(gal_array=gal_data, noise_array=noise_data)
        out1 = task.find_peaks(
            smooth_data,
            fthres=1.0,
            pthres=pthres,
            pratio=pratio,
            bound=32,
            std_m00=std * scale**2.0,
            std_v=std * scale**2.0,
        )[0:30000]
        mtask = anacal.fpfs.FpfsImage(
            nx=64,
            ny=64,
            scale=scale,
            sigma_arcsec=sigma_as,
            klim=tt.klim / scale,
            psf_array=psf_data,
        )
        src1 = mtask.measure_sources(
            gal_data,
            filter_image=tt.bfunc,
            det=out1
        )
        del noise_data, out1, smooth_data, src1
        return

    print_mem(initial_memory_usage)
    t0 = time.time()
    func()
    t1 = time.time()
    print("Time: ", t1 - t0)

    peak_memory_usage = max(memory_usage(proc=(func,)))
    print("Peak Mem:", peak_memory_usage, "M")
    del task
    gc.collect()
    final_memory_usage = mem_used()
    print_mem(final_memory_usage - initial_memory_usage)
    return


if __name__ == "__main__":
    test_detect()
