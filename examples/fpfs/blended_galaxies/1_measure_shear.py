import anacal
import fitsio
import numpy as np

fpfs_config = anacal.fpfs.FpfsConfig()
gal_array = fitsio.read("image-00000_g1-0_rot0_i.fits")
psf_array = fitsio.read("PSF_Fixed.fits")
pixel_scale = 0.2
noise_variance = 0.23
noise_array = None
cov_matrix = None
coords = None
coords, outcome = anacal.fpfs.process_image(
    fpfs_config,
    gal_array,
    psf_array,
    pixel_scale,
    noise_variance,
    noise_array,
    cov_matrix,
    coords,
)

print(np.sum(outcome[:, 0]) / np.sum(outcome[:, 1]))
