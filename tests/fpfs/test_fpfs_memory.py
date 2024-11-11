import os

import anacal
import fitsio
import numpy as np
from memory_profiler import memory_usage

from .. import mem_used, print_mem

data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../examples/fpfs/blended_galaxies",
)
gal_array = np.asarray(
    fitsio.read(os.path.join(data_dir, "image-00000_g1-0_rot0_i.fits")),
    dtype=np.float64,
)
psf_array = np.asarray(
    fitsio.read(os.path.join(data_dir, "PSF_Fixed.fits")),
    dtype=np.float64,
)


def func():
    fpfs_config = anacal.fpfs.FpfsConfig(
        sigma_arcsec=0.52,  # The first measurement scale (also for detection)
        sigma_arcsec1=0.45,  # The second measurement scale
        sigma_arcsec2=0.60,  # The third measurement scale
    )
    mag_zero = 30.0
    pixel_scale = 0.2
    noise_variance = 0.23**2.0
    noise_array = None
    detection = None
    out = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=mag_zero,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        noise_variance=noise_variance,
        noise_array=noise_array,
        detection=detection,
    )

    e1 = out["fpfs_w"] * out["fpfs_e1"]
    e1g1 = (
        out["fpfs_dw_dg1"] * out["fpfs_e1"]
        + out["fpfs_w"] * out["fpfs_de1_dg1"]
    )
    print(np.sum(e1) / np.sum(e1g1))

    e1 = out["fpfs_w"] * out["fpfs1_e1"]
    e1g1 = (
        out["fpfs_dw_dg1"] * out["fpfs1_e1"]
        + out["fpfs_w"] * out["fpfs1_de1_dg1"]
    )
    print(np.sum(e1) / np.sum(e1g1))

    e1 = out["fpfs_w"] * out["fpfs2_e1"]
    e1g1 = (
        out["fpfs_dw_dg1"] * out["fpfs2_e1"]
        + out["fpfs_w"] * out["fpfs2_de1_dg1"]
    )
    print(np.sum(e1) / np.sum(e1g1))
    del out, fpfs_config
    return


def test_memory():
    print("")
    print("Initial Mem:")
    initial_memory_usage = mem_used()
    print_mem(initial_memory_usage)
    func()
    peak_memory_usage = max(memory_usage(proc=(func,)))
    print("Peak Mem:", peak_memory_usage, "M")

    final_memory_usage = mem_used()
    print("Additional Mem:")
    print_mem(final_memory_usage - initial_memory_usage)
    return


if __name__ == "__main__":
    test_memory()
