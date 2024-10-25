# Image Tasks of Shapelet Based Measurements
#
# python lib
import numpy as np
import numpy.lib.recfunctions as rfn
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from . import BasePsf, FpfsImage, mask_galaxy_image, measure_fpfs
from .base import FpfsKernel

npix_patch = 256
npix_overlap = 64
npix_default = 64

std_m00_30 = 1.6
std_r2_30 = 3.2
std_v_30 = 0.6


class FpfsTask(FpfsKernel):
    """A base class for measurement

    Args:
    npix (int): number of pixels in a postage stamp
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    noise_variance (float): variance of image noise
    kmax (float | None): maximum k
    psf_array (ndarray): an average PSF image [default: None]
    kmax_thres (float): the tuncation threshold on Gaussian [default: 1e-20]
    do_detection (bool): whether compute detection kernel
    bound (int): minimum distance to boundary [default: 0]
    verbose (bool): whether print out INFO
    """

    def __init__(
        self,
        *,
        npix: int,
        pixel_scale: float,
        sigma_arcsec: float,
        noise_variance: float = -1,
        kmax: float | None = None,
        psf_array: NDArray | None = None,
        kmax_thres: float = 1e-20,
        do_detection: bool = True,
        bound: int = 0,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            npix=npix,
            pixel_scale=pixel_scale,
            sigma_arcsec=sigma_arcsec,
            kmax=kmax,
            psf_array=psf_array,
            kmax_thres=kmax_thres,
            do_detection=do_detection,
            verbose=verbose,
        )
        self.prepare_fpfs_bases()
        self.mtask = FpfsImage(
            nx=self.npix,
            ny=self.npix,
            scale=self.pixel_scale,
            sigma_arcsec=self.sigma_arcsec,
            klim=self.kmax / self.pixel_scale,
            psf_array=self.psf_array,
            use_estimate=True,
        )

        if self.do_detection:
            self.dtask = FpfsImage(
                nx=npix_patch,
                ny=npix_patch,
                scale=self.pixel_scale,
                sigma_arcsec=self.sigma_arcsec,
                klim=self.kmax / self.pixel_scale,
                psf_array=self.psf_array,
                use_estimate=True,
                npix_overlap=npix_overlap,
                bound=bound,
            )
            if not noise_variance > 0:
                raise ValueError("Noise variance should be positive")
            self.prepare_covariance(variance=noise_variance)
        else:
            self.dtask = None

        return

    def detect(
        self,
        *,
        gal_array: NDArray,
        fthres: float,
        pthres: float,
        noise_array: NDArray | None = None,
        mask_array: NDArray | None = None,
        star_cat: NDArray | None = None,
    ) -> NDArray:
        """This function detects galaxy from image

        Args:
        gal_array (NDArray): galaxy image data
        fthres (float): flux threshold
        pthres (float): peak threshold
        noise_array (NDArray|None): pure noise image
        mask_array (NDArray|None): mask image
        star_cat (NDArray|None): bright star catalog

        Returns:
        (NDArray): galaxy detection catalog
        """
        if mask_array is not None:
            # Set the value inside star mask to zero
            mask_galaxy_image(gal_array, mask_array, True, star_cat)
            if noise_array is not None:
                # Also do it for pure noise image
                mask_galaxy_image(noise_array, mask_array, False, star_cat)
        assert self.dtask is not None
        return self.dtask.detect_source(
            gal_array=gal_array,
            fthres=fthres,
            pthres=pthres,
            std_m00=self.std_m00 * self.pixel_scale**2.0,
            std_v=self.std_v * self.pixel_scale**2.0,
            noise_array=noise_array,
            mask_array=mask_array,
        )

    def run_psf_array(
        self,
        *,
        gal_array: NDArray,
        psf_array: NDArray,
        det: NDArray | None = None,
        noise_array: NDArray | None = None,
    ) -> tuple[NDArray, NDArray | None]:
        """This function measure galaxy shapes at the position of the detection
        using PSF image data

        Args:
        gal_array (NDArray): galaxy image data
        psf_array (NDArray): psf image data
        det (list|None): detection catalog
        noise_array (NDArray | None): noise image data [default: None]

        Returns:
        src_g (NDArray): source measurement catalog
        src_n (NDArray): noise measurement catalog
        """
        # self.logger.warning("Input PSF is array")
        src_g = self.mtask.measure_source(
            gal_array=gal_array,
            filter_image=self.bfunc_use,
            psf_array=psf_array,
            det=det,
            do_rotate=False,
        )
        if noise_array is not None:
            src_n = self.mtask.measure_source(
                gal_array=noise_array,
                filter_image=self.bfunc_use,
                psf_array=psf_array,
                det=det,
                do_rotate=True,
            )
            src_g = src_g + src_n
        else:
            src_n = None
        return src_g, src_n

    def run_psf_python(
        self,
        gal_array: NDArray,
        psf_obj: BasePsf,
        det: NDArray,
        noise_array: NDArray | None = None,
    ) -> tuple[NDArray, NDArray | None]:
        """This function measure galaxy shapes at the position of the detection
        using PSF image data

        Args:
        gal_array (NDArray): galaxy image data
        psf_obj (BasePsf): PSF object in python
        noise_array (NDArray | None): noise image data [default: None]
        det (list|None): detection catalog

        Returns:
        src_g (NDArray): source measurement catalog
        src_n (NDArray): noise measurement catalog
        """
        # self.logger.warning("Input PSF is python object")
        det_dtype = det.dtype
        src_g = []
        src_n = []
        for _d in det:
            this_psf_array = psf_obj.draw(x=_d["x"], y=_d["y"])
            # TODO: remove det_array
            det_array = np.array([_d], dtype=det_dtype)
            srow = self.mtask.measure_source(
                gal_array=gal_array,
                filter_image=self.bfunc_use,
                psf_array=this_psf_array,
                det=det_array,
                do_rotate=False,
            )[0]
            if noise_array is not None:
                nrow = self.mtask.measure_source(
                    gal_array=noise_array,
                    filter_image=self.bfunc_use,
                    psf_array=this_psf_array,
                    det=det_array,
                    do_rotate=True,
                )[0]
                srow = srow + nrow
                src_n.append(nrow)
            src_g.append(srow)
        if len(src_n) == 0:
            src_n = None
        else:
            assert len(src_n) == len(src_g)
            src_n = np.array(src_n)
        src_g = np.array(src_g)
        return src_g, src_n

    def run_psf_cpp(
        self,
        gal_array: NDArray,
        psf_obj: BasePsf,
        noise_array: NDArray | None = None,
        det: NDArray | None = None,
    ) -> tuple[NDArray, NDArray | None]:
        """This function measure galaxy shapes at the position of the detection
        using PSF model with spacial variation

        Args:
        gal_array (NDArray): galaxy image data
        psf_obj (BasePsf): psf object in cpp
        noise_array (NDArray | None): noise image data [default: None]
        det (list|None): detection catalog

        Returns:
        src_g (NDArray): source measurement catalog
        src_n (NDArray): noise measurement catalog
        """

        # self.logger.warning("Input PSF is cpp object")
        src_g = self.mtask.measure_source(
            gal_array=gal_array,
            filter_image=self.bfunc_use,
            psf_obj=psf_obj,
            det=det,
            do_rotate=False,
        )
        if noise_array is not None:
            src_n = self.mtask.measure_source(
                gal_array=noise_array,
                filter_image=self.bfunc_use,
                psf_obj=psf_obj,
                det=det,
                do_rotate=True,
            )
            src_g = src_g + src_n
        else:
            src_n = None
        return src_g, src_n

    def run(
        self,
        gal_array: NDArray,
        psf: BasePsf | NDArray,
        det: NDArray | None = None,
        noise_array: NDArray | None = None,
    ):
        """This function measure galaxy shapes at the position of the detection

        Args:
        gal_array (NDArray): galaxy image data
        det (NDArray): detection catalog
        psf (BasePsf | NDArray): psf image data or psf model
        noise_array (NDArray | None): noise image data [default: None]

        Returns:
        (NDArray): galaxy measurement catalog
        """
        if isinstance(psf, np.ndarray):
            src_g, src_n = self.run_psf_array(
                gal_array=gal_array,
                psf_array=psf,
                noise_array=noise_array,
                det=det,
            )
        elif isinstance(psf, BasePsf):
            assert det is not None
            if psf.crun:
                # For the case PSF is a C++ object
                src_g, src_n = self.run_psf_cpp(
                    gal_array=gal_array,
                    psf_obj=psf,
                    noise_array=noise_array,
                    det=det,
                )
            else:
                # For the case PSF is a Python object
                src_g, src_n = self.run_psf_python(
                    gal_array=gal_array,
                    psf_obj=psf,
                    noise_array=noise_array,
                    det=det,
                )

        else:
            raise RuntimeError("psf does not have a correct type")
        src_g = rfn.unstructured_to_structured(
            arr=src_g,
            dtype=self.dtype,
        )
        if src_n is not None:
            src_n = rfn.unstructured_to_structured(arr=src_n, dtype=self.dtype)

        return {
            "data": src_g,
            "noise": src_n,
        }


class FpfsConfig(BaseModel):
    npix: int = Field(
        default=64,
        description="""size of the stamp before Fourier Transform
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
    do_compute_detect_weight: bool = True,
):
    """Run measurement algorithms on the input exposure, and optionally
    populate the resulting catalog with extra information.

    Args:
    fpfs_config (FpfsConfig):  configuration object
    pixel_scale (float): pixel scale in arcsec
    noise_variance (float): variance of image noise
    mag_zero (float): magnitude zero point
    do_detection (bool): whether compute detection kernel
    gal_array (NDArray[float64]): galaxy exposure array
    psf_array (ndarray): an average PSF image
    noise_array (NDArray | None): pure noise array [default: None]
    mask_array (NDArray | None): mask array (1 for masked) [default: None]
    star_catalog (NDArray | None): bright star catalog [default: None]
    detection (NDArray | None): detection catalog [default: None]
    psf_object (BasePsf | None): PSF object [default: None]
    do_compute_detect_weight (bool): whether to compute detection weight

    Returns:
    (NDArray) FPFS catalog
    """
    if psf_object is None:
        psf_object = psf_array

    out_list = []
    ratio = 1.0 / (10 ** ((30 - mag_zero) / 2.5))
    std_r2 = std_r2_30 * ratio
    std_v = std_v_30 * ratio
    ftask = FpfsTask(
        npix=fpfs_config.npix,
        pixel_scale=pixel_scale,
        sigma_arcsec=fpfs_config.sigma_arcsec,
        noise_variance=noise_variance,
        psf_array=psf_array,
        kmax_thres=fpfs_config.kmax_thres,
        do_detection=True,
        bound=fpfs_config.bound,
    )

    std_m00 = ftask.std_m00

    ftask.std_r2 = std_r2
    ftask.std_v = std_v
    std_m00 = ftask.std_m00
    m00_min = fpfs_config.snr_min * std_m00
    fpfs_c0 = fpfs_config.c0 * std_m00

    if do_compute_detect_weight or (detection is None):
        if detection is None:
            detection = ftask.detect(
                gal_array=gal_array,
                fthres=fpfs_config.fthres,
                pthres=fpfs_config.pthres,
                noise_array=noise_array,
                mask_array=mask_array,
                star_cat=star_catalog,
            )
        out_list.append(detection)

        if do_compute_detect_weight:
            # Measurement Tasks
            src = ftask.run(
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
                x_array=src["data"],
                y_array=src["noise"],
            )
            del src
            out_list.append(meas)

    del ftask

    if fpfs_config.sigma_arcsec1 > 0:
        ftask = FpfsTask(
            npix=fpfs_config.npix,
            pixel_scale=pixel_scale,
            sigma_arcsec=fpfs_config.sigma_arcsec1,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            do_detection=False,
        )
        src = ftask.run(
            gal_array=gal_array,
            psf=psf_object,
            det=detection,
            noise_array=noise_array,
        )
        meas1 = measure_fpfs(
            C0=fpfs_c0,
            x_array=src["data"],
            y_array=src["noise"],
        )
        del src, ftask
        map_dict = {name: name + "_1" for name in meas1.dtype.names}
        out_list.append(rfn.rename_fields(meas1, map_dict))

    if fpfs_config.sigma_arcsec2 > 0:
        ftask = FpfsTask(
            npix=fpfs_config.npix,
            pixel_scale=pixel_scale,
            sigma_arcsec=fpfs_config.sigma_arcsec2,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            do_detection=False,
        )
        src = ftask.run(
            gal_array=gal_array,
            psf=psf_object,
            det=detection,
            noise_array=noise_array,
        )
        meas2 = measure_fpfs(
            C0=fpfs_c0,
            x_array=src["data"],
            y_array=src["noise"],
        )
        del src, ftask
        map_dict = {name: name + "_2" for name in meas2.dtype.names}
        out_list.append(rfn.rename_fields(meas2, map_dict))

    return rfn.merge_arrays(
        out_list,
        flatten=True,
        usemask=False,
    )
