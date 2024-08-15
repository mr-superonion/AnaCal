import numpy as np
import numpy.lib.recfunctions as rfn
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from .._anacal.fpfs import (
    FpfsImage,
    fpfs_cut_sigma_ratio,
    fpfs_det_sigma2,
    fpfs_pnr,
)
from .._anacal.image import Image
from .._anacal.mask import mask_galaxy_image
from .._anacal.psf import BasePsf
from . import base, table
from .ctask import CatalogTask
from .itask import FpfsDetect, FpfsMeasure, FpfsNoiseCov

__all__ = [
    "base",
    "table",
    "FpfsImage",
    "FpfsNoiseCov",
    "FpfsDetect",
    "FpfsMeasure",
    "FpfsConfig",
    "CatalogTask",
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
    sigma_arcsec2: float = Field(
        default=-1,
        description="""Smoothing scale of the second shapelet kernel.
        """,
    )
    pthres: float = Field(
        default=0.12,
        description="""Detection threshold (peak identification) for the
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


def detect_sources(
    *,
    fpfs_config: FpfsConfig,
    pixel_scale: float,
    cov_matrix: table.Covariance,
    gal_array: NDArray,
    psf_array: NDArray,
    mask_array: NDArray | None = None,
    star_catalog: NDArray | None = None,
    noise_array: NDArray | None = None,
    mag_zero: float = 30.0,
    **kwargs,
):
    dtask = FpfsDetect(
        mag_zero=mag_zero,
        pixel_scale=pixel_scale,
        sigma_arcsec=fpfs_config.sigma_arcsec,
        cov_matrix=cov_matrix,
        det_nrot=4,
        psf_array=psf_array,
        kmax_thres=fpfs_config.kmax_thres,
        bound=fpfs_config.bound,
    )
    detection = dtask.run(
        gal_array=gal_array,
        fthres=8.0,
        pthres=fpfs_config.pthres,
        noise_array=noise_array,
        mask_array=mask_array,
        star_cat=star_catalog,
    )
    return detection


def _measure_one_gal(
    *,
    gal_array,
    this_psf_array,
    det_array,
    noise_array,
    mtask,
    ctask,
):
    srow, nrow = mtask.run_single_psf(
        gal_array=gal_array,
        psf_array=this_psf_array,
        det=det_array,
        noise_array=noise_array,
    )
    srow = srow[0]
    if nrow is None:
        nrow = 0.0
    else:
        nrow = nrow[0]
    return tuple(ctask._run(srow, nrow))


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
    **kwargs,
):
    """Run measurement algorithms on the input exposure, and optionally
    populate the resulting catalog with extra information.

    Args:

    Returns:
    """

    cov_task = FpfsNoiseCov(
        mag_zero=mag_zero,
        pixel_scale=pixel_scale,
        sigma_arcsec=fpfs_config.sigma_arcsec,
        norder=fpfs_config.norder,
        det_nrot=4,
        psf_array=psf_array,
        kmax_thres=fpfs_config.kmax_thres,
    )
    # Since we have additional layer of noise, we need to multiply by two
    cov_matrix = cov_task.measure(variance=noise_variance * 2.0)
    del cov_task

    if detection is None:
        detection = detect_sources(
            fpfs_config=fpfs_config,
            pixel_scale=pixel_scale,
            cov_matrix=cov_matrix,
            gal_array=gal_array,
            psf_array=psf_array,
            mask_array=mask_array,
            star_catalog=star_catalog,
            noise_array=noise_array,
            mag_zero=mag_zero,
        )

    # Measurement Tasks
    # First Measurement comes with detection
    mtask_d = FpfsMeasure(
        mag_zero=mag_zero,
        pixel_scale=pixel_scale,
        sigma_arcsec=fpfs_config.sigma_arcsec,
        psf_array=psf_array,
        kmax_thres=fpfs_config.kmax_thres,
        norder=fpfs_config.norder,
        det_nrot=4,
    )
    if fpfs_config.sigma_arcsec2 > 0.0:
        mtask_m = FpfsMeasure(
            mag_zero=mag_zero,
            pixel_scale=pixel_scale,
            sigma_arcsec=fpfs_config.sigma_arcsec2,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            norder=fpfs_config.norder,
            det_nrot=-1,
        )
    else:
        mtask_m = None

    # Catalog Task
    cat_task = CatalogTask(
        norder=fpfs_config.norder,
        det_nrot=4,
        cov_matrix=cov_matrix,
    )
    cat_task.update_parameters(
        snr_min=fpfs_config.snr_min,
        r2_min=fpfs_config.r2_min,
        c0=fpfs_config.c0,
        pthres=fpfs_config.pthres,
    )
    assert cat_task.det_task is not None

    cat_task.det_task.pixel_scale = pixel_scale
    cat_task.det_task.sigma_arcsec = fpfs_config.sigma_arcsec
    out_dtype = cat_task.det_task.dtype + cat_task.meas_task.dtype
    if (psf_object is not None) and (not psf_object.crun):
        meas = []
        det_dtype = detection.dtype
        for det in detection:
            this_psf_array = psf_object.draw(x=det["x"], y=det["y"])
            det_array = np.array([det], dtype=det_dtype)
            meas_row = _measure_one_gal(
                gal_array=gal_array,
                this_psf_array=this_psf_array,
                det_array=det_array,
                noise_array=noise_array,
                mtask=mtask_d,
                ctask=cat_task.det_task,
            )
            if mtask_m is not None:
                meas_row = meas_row + _measure_one_gal(
                    gal_array=gal_array,
                    this_psf_array=this_psf_array,
                    det_array=det_array,
                    noise_array=noise_array,
                    mtask=mtask_m,
                    ctask=cat_task.meas_task,
                )
            meas.append(meas_row)
        meas = np.array(meas, dtype=out_dtype)
    else:
        src1 = mtask_d.run(
            gal_array=gal_array,
            psf=psf_array,
            det=detection,
            noise_array=noise_array,
        )
        if mtask_m is not None:
            src2 = mtask_m.run(
                gal_array=gal_array,
                psf=psf_array,
                det=detection,
                noise_array=noise_array,
            )
        else:
            src2 = None
        meas = cat_task.run(catalog=src1, catalog2=src2)

    return rfn.merge_arrays(
        [detection, meas],
        flatten=True,
        usemask=False,
    )
