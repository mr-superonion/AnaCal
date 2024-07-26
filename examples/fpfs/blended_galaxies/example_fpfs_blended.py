import os

import anacal
import fitsio
import numpy as np

data_dir = os.path.dirname(os.path.abspath(__file__))

fpfs_config = anacal.fpfs.FpfsConfig(
    sigma_arcsec=0.52,  # The first measurement scale (also for detection)
    sigma_arcsec2=0.45,  # The second measurement scale
)
gal_array = fitsio.read(os.path.join(data_dir, "image-00000_g1-0_rot0_i.fits"))
psf_array = fitsio.read(os.path.join(data_dir, "PSF_Fixed.fits"))
pixel_scale = 0.2
noise_variance = 0.23
noise_array = None
coords = None
out = anacal.fpfs.process_image(
    fpfs_config=fpfs_config,
    gal_array=gal_array,
    psf_array=psf_array,
    pixel_scale=pixel_scale,
    noise_variance=noise_variance,
    noise_array=noise_array,
    coords=coords,
)

e1 = out["w"] * out["e1"]
e1g1 = out["w_g1"] * out["e1"] + out["w"] * out["e1_g1"]
print(np.sum(e1) / np.sum(e1g1))

e1 = out["w"] * out["e1_2"]
e1g1 = out["w_g1"] * out["e1_2"] + out["w"] * out["e1_g1_2"]
print(np.sum(e1) / np.sum(e1g1))
