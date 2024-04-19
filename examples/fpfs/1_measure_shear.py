import fitsio
import anacal
import numpy as np
from dataclasses import dataclass


@dataclass
class FpfsConfig:
    rcut: int
    psf_rcut: int
    noise_rev: bool
    nord: int
    det_nrot: int
    klim_thres: float
    bound: int
    sigma_arcsec: float
    pratio: float
    pthres: float
    pthres2: float
    snr_min: float
    r2_min: float
    c0: float
    c2: float
    alpha: float
    beta: float


gal_array = fitsio.read()
psf_array = fitsio.read()
noise_variance = 0.23
pixel_scale = 0.2
fpfs_config = FpfsConfig()
seed = 1

ny, nx = gal_array.shape
if fpfs_config.noise_rev:
    noise_array = np.random.RandomState(seed).normal(
        scale=np.sqrt(noise_variance),
        size=(ny, nx),
    )
else:
    noise_array = None
# Shapelet Covariance matrix
noise_task = anacal.fpfs.FpfsNoiseCov(
    psf_array=psf_array,
    pixel_scale=pixel_scale,
    sigma_arcsec=fpfs_config.sigma_arcsec,
    nord=fpfs_config.nord,
    det_nrot=fpfs_config.det_nrot,
    klim_thres=fpfs_config.klim_thres,
)
cov_matrix = noise_task.measure(variance=noise_variance)
del noise_task

# Detection
dtask = anacal.fpfs.FpfsDetect(
    nx=ny,
    ny=nx,
    psf_array=psf_array,
    pix_scale=pixel_scale,
    sigma_arcsec=fpfs_config.sigma_arcsec,
    cov_matrix=cov_matrix,
    det_nrot=fpfs_config.det_nrot,
    klim_thres=fpfs_config.klim_thres,
)
coords = dtask.run(
    gal_array=gal_array,
    fthres=8.5,
    pthres=fpfs_config.pthres,
    pratio=fpfs_config.pratio,
    pthres2=fpfs_config.pthres2,
    bound=fpfs_config.bound,
    noise_array=noise_array,
)
del dtask

mtask = anacal.fpfs.FpfsMeasure(
    psf_array=psf_array,
    pix_scale=pixel_scale,
    sigma_arcsec=fpfs_config.sigma_arcsec,
    det_nrot=fpfs_config.det_nrot,
    klim_thres=fpfs_config.klim_thres,
)
src = mtask.run(gal_array=gal_array, det=coords)
if noise_array is not None:
    noise_src = mtask.run(noise_array, det=coords, do_rotate=True)
    src = src + noise_src
else:
    noise_src = None
sel = (src[:, mtask.di["m00"]] + src[:, mtask.di["m20"]]) > 1e-5
coords = np.array(coords)[sel]
src = src[sel]
if noise_src is not None:
    noise_src = noise_src[sel]
del mtask, sel

# Catalog

cat_obj = anacal.fpfs.FpfsCatalog(
    cov_mat=cov_matrix,
    snr_min=fpfs_config.snr_min,
    r2_min=fpfs_config.r2_min,
    c0=fpfs_config.c0,
    c2=fpfs_config.c2,
    alpha=fpfs_config.alpha,
    beta=fpfs_config.beta,
    pthres=fpfs_config.pthres,
    pratio=fpfs_config.pratio,
    pthres2=fpfs_config.pthres2,
    det_nrot=fpfs_config.det_nrot,
)

e1, e1_res = cat_obj.measure_g1_renoise(src, noise_src)
