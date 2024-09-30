import numpy as np
import numpy.lib.recfunctions as rfn
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from .._anacal.fpfs import (
    fpfs_cut_sigma_ratio,
    fpfs_det_sigma2,
    fpfs_pnr,
    FpfsImage,
    measure_fpfs,
    measure_fpfs_shape,
    measure_fpfs_wdet,
    measure_fpfs_wsel,
    measure_shapelets_dg,
)
from .._anacal.image import Image
from .._anacal.mask import mask_galaxy_image
from .._anacal.psf import BasePsf
from .base import FpfsKernel
from .itask import FpfsDetect, FpfsMeasure

__all__ = [
    "base",
    "FpfsImage",
    "FpfsDetect",
    "FpfsMeasure",
    "FpfsConfig",
]


class FpfsConfig(BaseModel):
    npix: int = Field(
        default=64,
        description="""size of the stamp before Fourier Transform
        """,
    )
    norder: int = Field(
        default=4,
        description="""Maximum radial number `n` to use for the shapelet basis
        """,
    )
    kmax_thres: float = Field(
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
    sigma_arcsec1: float = Field(
        default=-1,
        description="""Smoothing scale of the second shapelet kernel.
        """,
    )
    sigma_arcsec2: float = Field(
        default=-1,
        description="""Smoothing scale of the third shapelet kernel.
        """,
    )
    pthres: float = Field(
        default=0.12,
        description="""Detection threshold (peak identification) for the
        pooling.
        """,
    )
    fthres: float = Field(
        default=8.0,
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


def detect_sources(
    *,
    fpfs_config: FpfsConfig,
    kernel: FpfsKernel,
    gal_array: NDArray,
    mask_array: NDArray | None = None,
    star_catalog: NDArray | None = None,
    noise_array: NDArray | None = None,
):
    """Detect sources from exposure
    fpfs_config (FpfsConfig):  configuration object
    kernel (FpfsKernel): kernel object
    gal_array (NDArray[float64]): galaxy exposure array
    mask_array (NDArray[int]): mask array (1 for masked)
    star_catalog (NDArray[BrightStar]): bright star catalog
    noise_array (NDArray[float64]): pure noise array
    """
    dtask = FpfsDetect(
        kernel=kernel,
        bound=fpfs_config.bound,
    )
    detection = dtask.run(
        gal_array=gal_array,
        fthres=fpfs_config.fthres,
        pthres=fpfs_config.pthres,
        noise_array=noise_array,
        mask_array=mask_array,
        star_cat=star_catalog,
    )
    return detection


def process_image(
    *,
    fpfs_config: FpfsConfig,
    pixel_scale: float,
    noise_variance: float,
    mag_zero: float,
    gal_array: NDArray,
    psf_array: NDArray,
    noise_array: NDArray | None = None,
    mask_array: NDArray | None = None,
    star_catalog: NDArray | None = None,
    detection: NDArray | None = None,
    psf_object: BasePsf | None = None,
):
    """Run measurement algorithms on the input exposure, and optionally
    populate the resulting catalog with extra information.

    Args:

    Returns:
    """

    out_list = []
    kernel = FpfsKernel(
        npix=fpfs_config.npix,
        pixel_scale=pixel_scale,
        sigma_arcsec=fpfs_config.sigma_arcsec,
        psf_array=psf_array,
        kmax_thres=fpfs_config.kmax_thres,
        compute_detect_kernel=True,
    )
    kernel.prepare_fpfs_bases()
    kernel.prepare_covariance(variance=noise_variance * 2.0)

    fpfs_c0 = fpfs_config.c0 * kernel.cov_matrix.std_m00
    std_v = kernel.cov_matrix.std_v
    m00_min = fpfs_config.snr_min * kernel.cov_matrix.std_m00
    std_m00 = kernel.cov_matrix.std_m00
    std_r2 = kernel.cov_matrix.std_r2

    if detection is None:
        detection = detect_sources(
            fpfs_config=fpfs_config,
            kernel=kernel,
            gal_array=gal_array,
            mask_array=mask_array,
            star_catalog=star_catalog,
            noise_array=noise_array,
        )
    out_list.append(detection)

    if psf_object is None:
        psf_object = psf_array

    # Measurement Tasks
    meas_task = FpfsMeasure(
        kernel=kernel,
    )
    src_g, src_n = meas_task.run(
        gal_array=gal_array,
        psf=psf_object,
        det=detection,
        noise_array=noise_array,
    )
    meas = measure_fpfs(
        C0=fpfs_c0,
        std_v=std_v,
        pthres=fpfs_config.pthres,
        m00_min=m00_min,
        std_m00=std_m00,
        r2_min=fpfs_config.r2_min,
        std_r2=std_r2,
        x_array=src_g,
        y_array=src_n,
    )
    del kernel, meas_task, src_g, src_n
    out_list.append(meas)

    if fpfs_config.sigma_arcsec1 > 0:
        kernel = FpfsKernel(
            npix=fpfs_config.npix,
            pixel_scale=pixel_scale,
            sigma_arcsec=fpfs_config.sigma_arcsec1,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            compute_detect_kernel=False,
        )
        kernel.prepare_fpfs_bases()

        # Measurement Tasks
        meas_task = FpfsMeasure(
            kernel=kernel,
        )
        src_g, src_n = meas_task.run(
            gal_array=gal_array,
            psf=psf_object,
            det=detection,
            noise_array=noise_array,
        )
        meas1 = measure_fpfs(
            C0=fpfs_c0,
            x_array=src_g,
            y_array=src_n,
        )
        del kernel, meas_task, src_g, src_n
        map_dict = {name: name + "_1" for name in meas1.dtype.names}
        out_list.append(rfn.rename_fields(meas1, map_dict))

    if fpfs_config.sigma_arcsec2 > 0:
        kernel = FpfsKernel(
            npix=fpfs_config.npix,
            pixel_scale=pixel_scale,
            sigma_arcsec=fpfs_config.sigma_arcsec2,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            compute_detect_kernel=False,
        )
        kernel.prepare_fpfs_bases()

        # Measurement Tasks
        meas_task = FpfsMeasure(
            kernel=kernel,
        )
        src_g, src_n = meas_task.run(
            gal_array=gal_array,
            psf=psf_object,
            det=detection,
            noise_array=noise_array,
        )
        meas2 = measure_fpfs(
            C0=fpfs_c0,
            x_array=src_g,
            y_array=src_n,
        )
        del kernel, meas_task, src_g, src_n
        map_dict = {name: name + "_2" for name in meas2.dtype.names}
        out_list.append(rfn.rename_fields(meas2, map_dict))

    return rfn.merge_arrays(
        out_list,
        flatten=True,
        usemask=False,
    )
