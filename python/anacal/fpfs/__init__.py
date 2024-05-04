import numpy as np
from pydantic import BaseModel, Field

from .._anacal.fpfs import FpfsImage, fpfs_cut_sigma_ratio, fpfs_det_sigma2
from .._anacal.image import Image
from .._anacal.mask import mask_galaxy_image
from .._anacal.psf import BasePsf
from . import base
from .catalog import FpfsCatalog
from .task import FpfsDetect, FpfsMeasure, FpfsNoiseCov

__all__ = [
    "base",
    "FpfsImage",
    "FpfsDetect",
    "FpfsMeasure",
    "FpfsCatalog",
    "FpfsNoiseCov",
    "FpfsConfig",
]


def process_image(
    fpfs_config,
    gal_array,
    psf_array,
    pixel_scale,
    noise_variance,
    noise_array,
    cov_matrix,
    coords,
):
    # Preparing
    ngrid = fpfs_config.rcut * 2
    if not psf_array.shape == (ngrid, ngrid):
        raise ValueError("psf arry has a wrong shape")
    ny, nx = gal_array.shape
    if fpfs_config.noise_rev:
        if noise_array is None:
            raise ValueError("noise_rev is True, but no noise_array found")
    else:
        if noise_array is not None:
            raise ValueError("noise_rev is False, noise_array should be None")

    # Shapelet Covariance matrix
    if cov_matrix is None:
        if noise_variance <= 0:
            raise ValueError(
                "To enable detection, noise variance should be positive, ",
                "even though image is noiseless.",
            )
        noise_task = FpfsNoiseCov(
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
    if coords is None:
        dtask = FpfsDetect(
            nx=ny,
            ny=nx,
            psf_array=psf_array,
            pixel_scale=pixel_scale,
            sigma_arcsec=fpfs_config.sigma_arcsec,
            cov_matrix=cov_matrix,
            det_nrot=fpfs_config.det_nrot,
            klim_thres=fpfs_config.klim_thres,
        )
        coords = dtask.run(
            gal_array=gal_array,
            fthres=fpfs_config.fthres,
            pthres=fpfs_config.pthres,
            pratio=fpfs_config.pratio,
            pthres2=fpfs_config.pthres2,
            bound=fpfs_config.bound,
            noise_array=noise_array,
        )
        del dtask

    mtask = FpfsMeasure(
        psf_array=psf_array,
        pixel_scale=pixel_scale,
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
    ctask = FpfsCatalog(
        cov_matrix=cov_matrix,
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
    out = []
    if fpfs_config.gmeasure & 1:
        if not fpfs_config.force:
            out.append(ctask.measure_g1(src, noise_src))
        else:
            out.append(ctask.measure_g1_force(src, noise_src))
    if fpfs_config.gmeasure & 2:
        if not fpfs_config.force:
            out.append(ctask.measure_g2(src, noise_src))
        else:
            out.append(ctask.measure_g2_force(src, noise_src))
    out = np.hstack(out)
    return out


class FpfsConfig(BaseModel):
    force: bool = Field(
        default=False,
        description="""Whether this is a forced detection (selection). If true,
        we do not apply further detection and selection.
        """,
    )
    gmeasure: int = Field(
        default=1,
        description="""
        Which shear component to measure. '3' for both,  '1' for g1, '2' for g2
        """,
    )
    rcut: int = Field(
        default=32,
        description="""Galaxies are put into stamp before measurement, rcut
            is the radius of the cut
        """,
    )
    psf_rcut: int = Field(
        default=26,
        description="""Cut off radius for PSF.
        """,
    )
    noise_rev: bool = Field(
        default=False,
        description="""Whether do noise bias correction. The noise bias is
            corrected by adding noise to image to evaluate noise reponse.
        """,
    )
    nord: int = Field(
        default=4,
        description="""Maximum radial number `n` to use for the shapelet basis
        """,
    )
    det_nrot: int = Field(
        default=4,
        description="""Number of directions to calculate when detecting the
            peaks.
        """,
    )
    klim_thres: float = Field(
        default=1e-12,
        description="""The threshold used to define the upper limit of k we use
        in Fourier space.
        """,
    )
    bound: int = Field(
        default=35,
        description="""Boundary buffer length, the sources in the buffer reion
        are not counted.
        """,
    )
    sigma_arcsec: float = Field(
        default=0.52,
        description="""Smoothing scale of the shapelet and detection kernel.
        """,
    )
    pratio: float = Field(
        default=0.00,
        description="""Detection parameter (peak identification) for the first
        pooling.
        """,
    )
    pthres: float = Field(
        default=0.8,
        description="""Detection threshold (peak identification) for the first
        pooling.
        """,
    )
    pthres2: float = Field(
        default=0.12,
        description="""Detection threshold (peak identification) for the second
        pooling.
        """,
    )
    fthres: float = Field(
        default=8.5,
        description="""Detection threshold (minimum signal-to-noise ratio) for
        the first pooling.
        """,
    )
    snr_min: float = Field(
        default=12,
        description="""Minimum Signal-to-Noise Ratio.
        """,
    )
    r2_min: float = Field(
        default=0.1,
        description="""Minimum resolution.
        """,
    )
    c0: float = Field(
        default=5.0,
        description="""Weighting parameter for m00 for ellipticity definition.
        """,
    )
    c2: float = Field(
        default=22.0,
        description="""Weighting parameter for m20 for ellipticity definition.
        """,
    )
    alpha: float = Field(
        default=1.0,
        description="""Power parameter for m00 for ellipticity definition.
        """,
    )
    beta: float = Field(
        default=0.0,
        description="""Power parameter for m20 for ellipticity definition.
        """,
    )
