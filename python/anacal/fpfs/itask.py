# Image Tasks of Shapelet Based Measurements
#
# python lib
import numpy as np
from numpy.typing import NDArray

from . import BasePsf, FpfsImage, Image, mask_galaxy_image
from .base import FpfsKernel, name_s, name_d
import numpy.lib.recfunctions as rfn

npix_patch = 256
npix_overlap = 64
npix_default = 64


class FpfsMeasure(FpfsKernel):
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
        noise_variance: float,
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
            self.prepare_covariance(variance=noise_variance)
            self.std_m00 = self.cov_matrix.std_m00
            self.std_v = self.cov_matrix.std_v
        else:
            self.dtask = None

        self.mtask = FpfsImage(
            nx=self.npix,
            ny=self.npix,
            scale=self.pixel_scale,
            sigma_arcsec=self.sigma_arcsec,
            klim=self.kmax / self.pixel_scale,
            psf_array=self.psf_array,
            use_estimate=True,
        )
        return

    def detect(
        self,
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
            src_n = rfn.unstructured_to_structured(
                arr=src_n,
                dtype=self.dtype
            )

        return src_g, src_n
