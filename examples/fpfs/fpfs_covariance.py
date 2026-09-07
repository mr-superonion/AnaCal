import os

import anacal
import fitsio

# --- Configuration ---
pixel_scale = 0.2
noise_std = 0.6
noise_variance = noise_std**2.0
ngrid = 64

# --- PSF: pre-rendered Moffat (beta 2.5, fwhm 0.8), 64 x 64 at 0.2" ---
psf_array = fitsio.read(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "psf_moffat.fits")
)

ftask = anacal.fpfs.FpfsTask(
    npix=ngrid,
    psf_array=psf_array,
    pixel_scale=pixel_scale,
    sigma_shapelets=0.5374011,
    do_detection=True,
    noise_variance=noise_variance,
)

cov_matrix = ftask.prepare_covariance(variance=noise_variance)

print("Covariance matrix shape:", cov_matrix.shape)
print("Column names:", ftask.colnames)
print()

# Diagonal elements = variance of each mode
print("Std dev of each mode (ftask.std_modes):", ftask.std_modes)
print("Std dev of m00 (ftask.std_m00):", ftask.std_m00)
print()

# Print the full covariance matrix
# print("Covariance matrix:")
# print(np.array2string(cov_matrix, precision=6, suppress_small=True))
# print()
