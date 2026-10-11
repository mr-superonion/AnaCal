"""Memory of anacal.fpfs.process_image on the blended example scene
(examples/fpfs/blended_galaxies): the peak stays bounded and nothing
is left behind after the call.
"""

import os

import anacal
import fitsio
import numpy as np

from .. import mem_used, peak_mem_used, print_mem

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


def make_detection(pixel_scale, noise_variance):
    """Detect with the AnaCal detector (detector.h) and convert the
    positions to the ('y', 'x') pixel catalogue FPFS measures at --
    FPFS itself no longer detects.
    """
    det_task = anacal.task.Task(
        scale=pixel_scale,
        sigma_arcsec=0.5,
        snr_peak_min=12.0,
        omega_f=0.8,
        omega_v=0.04,
        mag_zero=30.0,
    )
    cells = anacal.geometry.get_cell_list(
        img_nx=gal_array.shape[1],
        img_ny=gal_array.shape[0],
        cell_nx=250,
        cell_ny=250,
        cell_overlap=80,
        scale=pixel_scale,
    )
    cat = det_task.process_image(
        np.asarray(gal_array, dtype=np.float32),
        psf_array,
        variance=noise_variance,
        cell_list=cells,
        do_measure=False,
    )
    detection = np.zeros(
        len(cat), dtype=[("y", np.float64), ("x", np.float64)]
    )
    detection["y"] = cat["x2_det"] / pixel_scale
    detection["x"] = cat["x1_det"] / pixel_scale
    # The differentiable selection weight and its shear response come from
    # the detector; FPFS carries no selection weight of its own.
    return detection, cat["wsel"], cat["dwsel_dg1"]


def func():
    fpfs_config = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=0.52,  # The first measurement scale
        sigma_shapelets2=0.60,  # The second measurement scale
    )
    mag_zero = 30.0
    pixel_scale = 0.2
    noise_variance = 0.23**2.0
    noise_array = None
    detection, wsel, dwsel_dg1 = make_detection(pixel_scale, noise_variance)
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

    # Shear estimate per measurement kernel, weighted by the detector's
    # differentiable selection weight so the selection response is
    # included: sum(w e) / sum(dw/dg e + w de/dg).  The scene is sheared
    # by g1 = -0.02 and blended, so only the sign and magnitude are
    # checked here.
    for kernel in ("fpfs1", "fpfs2"):
        e1 = out[f"{kernel}_e1"]
        e1g1 = out[f"{kernel}_de1_dg1"]
        g1 = np.sum(wsel * e1) / np.sum(dwsel_dg1 * e1 + wsel * e1g1)
        print(kernel, g1)
        assert abs(g1 / -0.02 - 1.0) < 0.1
    del out, fpfs_config


def test_memory():
    print("")
    print("Initial Mem:")
    initial_memory_usage = mem_used()
    print_mem(initial_memory_usage)
    func()
    peak_memory_usage = peak_mem_used(func)
    print("Peak Mem:", end=" ")
    print_mem(peak_memory_usage - initial_memory_usage)

    final_memory_usage = mem_used()
    print("Additional Mem:")
    print_mem(final_memory_usage - initial_memory_usage)
    # 1024 x 1024 image, ~3300 sources: measured ~230 MB peak above the
    # baseline and ~50 MB retained (allocator caches), a leak would grow
    assert peak_memory_usage - initial_memory_usage < 600 * 2**20
    assert final_memory_usage - initial_memory_usage < 200 * 2**20


if __name__ == "__main__":
    test_memory()
