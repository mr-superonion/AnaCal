import numpy as np
from numpy.typing import NDArray

from . import BasePsf, FpfsImage, Image, mask_galaxy_image
from .base import FpfsTask


class FpfsNoiseCov(FpfsTask):
    """A class to measure FPFS noise covariance of basis modes

    Args:
    psf_array (NDArray): an average PSF image used to initialize the task
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    sigma_arcsec_det (float|None): Detection kernel size [default: None]
    nord (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel
    klim_thres (float): the tuncation threshold on Gaussian [default: 1e-20]

    NOTE: The current version of Anacal.Fpfs only uses two elements of the
    covariance matrix. The full matrix will be useful in the future.
    """

    def __init__(
        self,
        psf_array: NDArray,
        pixel_scale: float,
        sigma_arcsec: float,
        sigma_arcsec_det: float | None = None,
        nord: int = 4,
        det_nrot: int = 4,
        klim_thres: float = 1e-20,
    ) -> None:
        super().__init__(
            psf_array=psf_array,
            pixel_scale=pixel_scale,
            sigma_arcsec=sigma_arcsec,
            sigma_arcsec_det=sigma_arcsec_det,
            nord=nord,
            det_nrot=det_nrot,
            klim_thres=klim_thres,
        )

        # Preparing PSF
        psf_f = np.fft.rfft2(psf_array)
        self.psf_pow = (np.abs(psf_f) ** 2.0).astype(np.float64)
        return

    def measure(self, variance, noise_pf=None):
        """Estimates covariance of measurement error

        Args:
        variance (float): noise variance
        noise_pf (NDArray|None): power spectrum (assuming homogeneous) of noise

        Return:
        cov_matrix (NDArray): covariance matrix of FPFS basis modes
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
        cov_matrix = np.tensordot(
            self.bfunc * (_w * noise_pf_deconv)[np.newaxis, :, :],
            np.conjugate(self.bfunc),
            axes=((1, 2), (1, 2)),
        ).real
        return cov_matrix


class FpfsDetect(FpfsTask):
    """A base class for measurement

    Args:
    nx (int): number of grids in x
    ny (int): number of grids in y
    psf_array (NDArray): an average PSF image used to initialize the task
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    sigma_arcsec_det (float|None): Detection kernel size [default: None]
    cov_matrix (NDArray): covariance matrix of Fpfs basis modes
    nord (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel [default: 8]
    klim_thres (float): the tuncation threshold on Gaussian [default: 1e-20]
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        psf_array: NDArray,
        cov_matrix: NDArray,
        pixel_scale: float,
        sigma_arcsec: float,
        sigma_arcsec_det: float | None = None,
        nord: int = 4,
        det_nrot: int = 4,
        klim_thres: float = 1e-20,
    ) -> None:
        super().__init__(
            psf_array=psf_array,
            sigma_arcsec=sigma_arcsec,
            sigma_arcsec_det=sigma_arcsec_det,
            pixel_scale=pixel_scale,
            nord=nord,
            det_nrot=det_nrot,
            klim_thres=klim_thres,
        )

        self.dtask = FpfsImage(
            nx=nx,
            ny=ny,
            scale=self.pixel_scale,
            sigma_arcsec=self.sigma_arcsec_det,
            klim=self.klim_det / self.pixel_scale,
            psf_array=psf_array,
            use_estimate=True,
        )
        self.nx = nx
        self.ny = ny

        self.std_m00, self.std_v = self.get_stds(cov_matrix)
        return

    def run(
        self,
        gal_array: NDArray,
        fthres: float,
        pthres: float,
        pratio: float,
        pthres2: float,
        bound: int,
        noise_array: NDArray | None = None,
        mask_array: NDArray | None = None,
        star_cat: NDArray | None = None,
    ) -> NDArray:
        """This function detects galaxy from image

        Args:
        gal_array (NDArray): galaxy image data
        fthres (float): flux threshold
        pthres (float): peak threshold
        pratio (float): peak flux ratio
        pthres2 (float): second pooling layer peak threshold
        bound (int): minimum distance to boundary
        noise_array (NDArray|None): pure noise image
        mask_array (NDArray|None): mask image
        star_cat (NDArray|None): bright star catalog

        Returns:
            galaxy detection catalog
        """

        if mask_array is not None:
            mask_galaxy_image(gal_array, mask_array, True, star_cat)
            if noise_array is not None:
                mask_galaxy_image(noise_array, mask_array, False, star_cat)

        ny, nx = gal_array.shape
        assert ny == self.ny
        assert nx == self.nx
        return self.dtask.detect_source(
            gal_array=gal_array,
            fthres=fthres,
            pthres=pthres,
            pratio=pratio,
            bound=bound,
            pthres2=pthres2,
            std_m00=self.std_m00,
            std_v=self.std_v,
            noise_array=noise_array,
            mask_array=mask_array,
        )


class FpfsMeasure(FpfsTask):
    """A base class for measurement

    Args:
    psf_array (NDArray): an average PSF image used to initialize the task
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    sigma_arcsec_det (float|None): Detection kernel size [default: None]
    nord (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel
    klim_thres (float): the tuncation threshold on Gaussian [default: 1e-20]
    """

    def __init__(
        self,
        psf_array: NDArray,
        pixel_scale: float,
        sigma_arcsec: float,
        sigma_arcsec_det: float | None = None,
        nord: int = 4,
        det_nrot: int = 4,
        klim_thres: float = 1e-20,
    ) -> None:
        super().__init__(
            psf_array=psf_array,
            pixel_scale=pixel_scale,
            sigma_arcsec=sigma_arcsec,
            sigma_arcsec_det=sigma_arcsec_det,
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
        return

    def run(
        self,
        gal_array: NDArray,
        psf: BasePsf | NDArray | None = None,
        det: NDArray | None = None,
        do_rotate: bool = False,
    ) -> NDArray:
        """This function detects galaxy from image

        Args:
        gal_array (NDArray): galaxy image data
        psf (BasePsf | NDArray | None): psf image data or psf model
        psf_obj (PSF object): reuturns PSF model according to position
        det (list|None): detection catalog
        do_rotate (bool): whether do rotation

        Returns:
        out (NDArray): galaxy measurement catalog
        """

        bfunc = np.transpose(self.bfunc, (1, 2, 0))
        if psf is None or isinstance(psf, np.ndarray):
            out = self.mtask.measure_source(
                gal_array=gal_array,
                filter_image=bfunc,
                psf_array=psf,
                det=det,
                do_rotate=do_rotate,
            )
        elif isinstance(psf, BasePsf):
            out = self.mtask.measure_source(
                gal_array=gal_array,
                filter_image=bfunc,
                psf_obj=psf,
                det=det,
                do_rotate=do_rotate,
            )
        else:
            raise RuntimeError("psf does not have a correct type")
        return out
