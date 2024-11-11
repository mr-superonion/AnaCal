import os

import anacal
import fitsio
import numpy as np

data_dir = os.path.dirname(os.path.abspath(__file__))

fpfs_config = anacal.fpfs.FpfsConfig(
    sigma_arcsec=0.52,  # The first measurement scale (also for detection)
    sigma_arcsec1=0.45,  # The second measurement scale
    sigma_arcsec2=0.55,  # The second measurement scale
)
gal_array = fitsio.read(os.path.join(data_dir, "image-00000_g1-0_rot0_i.fits"))
psf_array = fitsio.read(os.path.join(data_dir, "PSF_Fixed.fits"))
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

# base kernel scale
e1 = out["fpfs_w"] * out["fpfs_e1"]
e1g1 = (
    out["fpfs_dw_dg1"] * out["fpfs_e1"] + out["fpfs_w"] * out["fpfs_de1_dg1"]
)
print(np.sum(e1) / np.sum(e1g1))

# kernel 1
e1 = out["fpfs_w"] * out["fpfs1_e1"]
e1g1 = (
    out["fpfs_dw_dg1"] * out["fpfs1_e1"] + out["fpfs_w"] * out["fpfs1_de1_dg1"]
)
print(np.sum(e1) / np.sum(e1g1))

# kernel 2
e1 = out["fpfs_w"] * out["fpfs2_e1"]
e1g1 = (
    out["fpfs_dw_dg1"] * out["fpfs2_e1"] + out["fpfs_w"] * out["fpfs2_de1_dg1"]
)
print(np.sum(e1) / np.sum(e1g1))
