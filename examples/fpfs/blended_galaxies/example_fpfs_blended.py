import os

import anacal
import fitsio
import numpy as np

data_dir = os.path.dirname(os.path.abspath(__file__))

fpfs_config = anacal.fpfs.FpfsConfig(
    sigma_shapelets1=0.56,  # The first measurement scale
    sigma_shapelets2=0.60,  # The second measurement scale
)
gal_array = fitsio.read(os.path.join(data_dir, "image-00000_g1-0_rot0_i.fits"))
psf_array = fitsio.read(os.path.join(data_dir, "PSF_Fixed.fits"))
mag_zero = 30.0
pixel_scale = 0.2
noise_variance = 0.23**2.0
noise_array = None

# FPFS no longer detects internally: run the AnaCal detector
# (anacal.task.Task / detector.h) and convert its positions to the
# ('y', 'x') pixel catalogue FPFS measures at.
det_task = anacal.task.Task(
    scale=pixel_scale,
    sigma_arcsec=fpfs_config.sigma_shapelets1,
    snr_peak_min=12.0,
    omega_f=0.8,
    omega_v=0.04,
    mag_zero=mag_zero,
)
blocks = anacal.geometry.get_block_list(
    gal_array.shape[0],
    gal_array.shape[1],
    250,
    250,
    80,
    pixel_scale,
)
det_cat = det_task.process_image(
    np.asarray(gal_array, dtype=np.float32),
    psf_array,
    variance=noise_variance,
    block_list=blocks,
    do_measure=False,
)
detection = np.zeros(len(det_cat), dtype=[("y", "f8"), ("x", "f8")])
detection["y"] = det_cat["x2_det"] / pixel_scale
detection["x"] = det_cat["x1_det"] / pixel_scale

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

# Response ratios per measurement kernel (no selection weight here: the
# detection/selection weight lives in the AnaCal detector path).
# kernel 1
e1 = out["fpfs1_e1"]
e1g1 = out["fpfs1_de1_dg1"]
print(np.sum(e1) / np.sum(e1g1))

# kernel 2
e1 = out["fpfs2_e1"]
e1g1 = out["fpfs2_de1_dg1"]
print(np.sum(e1) / np.sum(e1g1))
