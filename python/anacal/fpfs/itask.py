# Image Tasks of Shapelet Based Measurements
#
# python lib
import numpy as np
from numpy.typing import NDArray

from . import BasePsf, FpfsImage, Image, mask_galaxy_image
from .base import ImgBase
from .table import Catalog, Covariance

npix_patch = 256
npix_overlap = 64


class FpfsNoiseCov(ImgBase):
    """A class to measure FPFS noise covariance of basis modes

    Args:
    mag_zero (float): magnitude zero point
    psf_array (NDArray): an average PSF image used to initialize the task
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    nord (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel
    klim_thres (float): the tuncation threshold on Gaussian [default: 1e-20]

    NOTE: The current version of Anacal.Fpfs only uses two elements of the
    covariance matrix. The full matrix will be useful in the future.
    """

    def __init__(
        self,
        mag_zero: float,
        psf_array: NDArray,
        pixel_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 4,
        klim_thres: float = 1e-20,
    ) -> None:
        super().__init__(
            mag_zero=mag_zero,
            psf_array=psf_array,
            pixel_scale=pixel_scale,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            det_nrot=det_nrot,
            klim_thres=klim_thres,
        )

        # Preparing PSF
        psf_f = np.fft.rfft2(psf_array)
        self.psf_pow = (np.abs(psf_f) ** 2.0).astype(np.float64)
        self.prepare_fpfs_bases()
        return

    def measure(
        self, variance: float, noise_pf: NDArray | None = None
    ) -> Covariance:
        """Estimates covariance of measurement error

        Args:
        variance (float): Noise variance
        noise_pf (NDArray | None): Power spectrum (assuming homogeneous) of
        noise

        Return:
        (Covariance): covariance matrix of FPFS basis modes
        """
        if noise_pf is not None:
            if noise_pf.shape == (self.ngrid, self.ngrid // 2 + 1):
                # rfft
                noise_pf = np.array(noise_pf, dtype=np.float64)
            elif noise_pf.shape == (self.ngrid, self.ngrid):
                # fft
                noise_pf = np.fft.ifftshift(noise_pf)
                noise_pf = np.array(
                    noise_pf[:, : self.ngrid // 2 + 1], dtype=np.float64
                )
            else:
                raise ValueError("noise power not in correct shape")
        else:
            ss = (self.ngrid, self.ngrid // 2 + 1)
            noise_pf = np.ones(ss)
        norm_factor = variance * self.ngrid**2.0 / noise_pf[0, 0]
        noise_pf = noise_pf * norm_factor

        img_obj = Image(nx=self.ngrid, ny=self.ngrid, scale=self.pixel_scale)
        img_obj.set_f(noise_pf)
        img_obj.deconvolve(
            psf_image=self.psf_pow,
            klim=self.klim / self.pixel_scale,
        )
        noise_pf_deconv = img_obj.draw_f().real

        _w = np.ones(self.psf_pow.shape) * 2.0
        _w[:, 0] = 1.0
        _w[:, -1] = 1.0
        cov_elems = (
            np.tensordot(
                self.bfunc * (_w * noise_pf_deconv)[np.newaxis, :, :],
                np.conjugate(self.bfunc),
                axes=((1, 2), (1, 2)),
            ).real
            / self.pixel_scale**4.0
        )
        return Covariance(
            array=cov_elems,
            mag_zero=self.mag_zero,
            nord=self.nord,
            det_nrot=self.det_nrot,
            pixel_scale=self.pixel_scale,
            sigma_arcsec=self.sigma_arcsec,
        )


class FpfsDetect(ImgBase):
    """A base class for measurement

    Args:
    mag_zero (float): magnitude zero point
    psf_array (NDArray): an average PSF image used to initialize the task
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Gaussian kernel size
    cov_matrix (Covariance): covariance matrix of Fpfs basis modes
    nord (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel [default: 8]
    klim_thres (float): the tuncation threshold on Gaussian [default: 1e-20]
    bound (int): minimum distance to boundary
    """

    def __init__(
        self,
        mag_zero: float,
        psf_array: NDArray,
        cov_matrix: Covariance,
        pixel_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 4,
        klim_thres: float = 1e-20,
        bound: int = 0,
    ) -> None:
        super().__init__(
            mag_zero=mag_zero,
            psf_array=psf_array,
            sigma_arcsec=sigma_arcsec,
            pixel_scale=pixel_scale,
            nord=nord,
            det_nrot=det_nrot,
            klim_thres=klim_thres,
        )

        self.dtask = FpfsImage(
            nx=npix_patch,
            ny=npix_patch,
            scale=self.pixel_scale,
            sigma_arcsec=self.sigma_arcsec,
            klim=self.klim / self.pixel_scale,
            psf_array=psf_array,
            use_estimate=True,
            npix_overlap=npix_overlap,
            bound=bound,
        )

        assert self.mag_zero == cov_matrix.mag_zero
        self.std_m00 = cov_matrix.std_m00
        self.std_v = cov_matrix.std_v
        return

    def run(
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

        # ny, nx = gal_array.shape
        # assert ny == self.ny
        # assert nx == self.nx
        return self.dtask.detect_source(
            gal_array=gal_array,
            fthres=fthres,
            pthres=pthres,
            std_m00=self.std_m00 * self.pixel_scale**2.0,
            std_v=self.std_v * self.pixel_scale**2.0,
            noise_array=noise_array,
            mask_array=mask_array,
        )


class FpfsMeasure(ImgBase):
    """A base class for measurement

    Args:
    mag_zero (float): magnitude zero point
    psf_array (NDArray): an average PSF image used to initialize the task
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    nord (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel
    klim_thres (float): the tuncation threshold on Gaussian [default: 1e-20]
    """

    def __init__(
        self,
        mag_zero: float,
        psf_array: NDArray,
        pixel_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 4,
        klim_thres: float = 1e-20,
    ) -> None:
        super().__init__(
            mag_zero=mag_zero,
            psf_array=psf_array,
            pixel_scale=pixel_scale,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            det_nrot=det_nrot,
            klim_thres=klim_thres,
        )
        self.mtask = FpfsImage(
            nx=self.ngrid,
            ny=self.ngrid,
            scale=self.pixel_scale,
            sigma_arcsec=self.sigma_arcsec,
            klim=self.klim / self.pixel_scale,
            psf_array=psf_array,
            use_estimate=True,
        )
        self.prepare_fpfs_bases()
        return

    def run(
        self,
        gal_array: NDArray,
        noise_array: NDArray | None = None,
        psf: BasePsf | NDArray | None = None,
        det: NDArray | None = None,
    ) -> Catalog:
        """This function measure galaxy shapes at the position of the detection

        Args:
        gal_array (NDArray): galaxy image data
        noise_array (NDArray | None): noise image data [default: None]
        psf (BasePsf | NDArray | None): psf image data or psf model
        det (list|None): detection catalog

        Returns:
        out (NDArray): galaxy measurement catalog
        """
        bfunc = np.transpose(self.bfunc, (1, 2, 0))
        if psf is None or isinstance(psf, np.ndarray):
            src_g = self.mtask.measure_source(
                gal_array=gal_array,
                filter_image=bfunc,
                psf_array=psf,
                det=det,
                do_rotate=False,
            )
            if noise_array is not None:
                src_n = self.mtask.measure_source(
                    gal_array=noise_array,
                    filter_image=bfunc,
                    psf_array=psf,
                    det=det,
                    do_rotate=True,
                )
                src_g = src_g + src_n
            else:
                src_n = None
        elif isinstance(psf, BasePsf):
            src_g = self.mtask.measure_source(
                gal_array=gal_array,
                filter_image=bfunc,
                psf_obj=psf,
                det=det,
                do_rotate=False,
            )
            if noise_array is not None:
                src_n = self.mtask.measure_source(
                    gal_array=noise_array,
                    filter_image=bfunc,
                    psf_obj=psf,
                    det=det,
                    do_rotate=True,
                )
                src_g = src_g + src_n
            else:
                src_n = None
        else:
            raise RuntimeError("psf does not have a correct type")
        return Catalog(
            array=src_g,
            noise=src_n,
            mag_zero=self.mag_zero,
            nord=self.nord,
            det_nrot=self.det_nrot,
            pixel_scale=self.pixel_scale,
            sigma_arcsec=self.sigma_arcsec,
        )
