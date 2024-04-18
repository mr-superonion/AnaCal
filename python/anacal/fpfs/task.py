import numpy as np
from numpy.typing import NDArray

from . import FpfsImage, Image
from .util import FpfsTask


class FpfsNoiseCov(FpfsTask):
    """A class to measure FPFS noise covariance of basis modes

    Args:
    psf_array (NDArray): an average PSF image used to initialize the task
    pix_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    nord (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel
    klim_thres (float): the tuncation threshold on Gaussian [default: 1e-20]

    NOTE: The current version of Anacal.Fpfs only uses two elements of the
    covariance matrix. The full matrix will be useful in the future.
    """

    def __init__(
        self,
        psf_array: NDArray,
        pix_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 4,
        klim_thres: float = 1e-20,
    ) -> None:
        super().__init__(
            psf_array=psf_array,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            pix_scale=pix_scale,
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

        img_obj = Image(nx=self.ngrid, ny=self.ngrid, scale=self.pix_scale)
        img_obj.set_f(noise_pf)
        img_obj.deconvolve(
            psf_image=self.psf_pow,
            klim=self.klim / self.pix_scale,
        )
        noise_pf_deconv = img_obj.draw_f().real

        _w = np.ones(self.psf_pow.shape) * 2.0
        _w[:, 0] = 1.0
        _w[:, -1] = 1.0
        cov_matrix = (
            np.tensordot(
                self.bfunc * (_w * noise_pf_deconv)[np.newaxis, :, :],
                np.conjugate(self.bfunc),
                axes=((1, 2), (1, 2)),
            ).real
            / self.pix_scale**4.0
        )
        return cov_matrix


class FpfsDetect(FpfsTask):
    """A base class for measurement

    Args:
    nx (int): number of grids in x
    ny (int): number of grids in y
    psf_array (NDArray): an average PSF image used to initialize the task
    pix_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
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
        pix_scale: float,
        sigma_arcsec: float,
        cov_matrix: NDArray,
        nord: int = 4,
        det_nrot: int = 4,
        klim_thres: float = 1e-20,
    ) -> None:
        super().__init__(
            psf_array=psf_array,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            pix_scale=pix_scale,
            det_nrot=det_nrot,
            klim_thres=klim_thres,
        )

        self.dtask = FpfsImage(
            nx=nx,
            ny=ny,
            scale=self.pix_scale,
            sigma_arcsec=self.sigma_arcsec,
            klim=self.klim / self.pix_scale,
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
        bound: int,
        noise_array: NDArray | None = None,
        wdet_cut: float = 0.0,
    ) -> NDArray:
        """This function detects galaxy from image

        Args:
        gal_array (NDArray): galaxy image data
        fthres (float): flux threshold
        pthres (float): peak threshold
        pratio (float): peak flux ratio
        bound (int): minimum distance to boundary
        noise_array (NDArray|None): pure noise image
        wdet_cut (float): lower limit of the detection weight

        Returns:
            galaxy detection catalog
        """
        ny, nx = gal_array.shape
        assert ny == self.ny
        assert nx == self.nx
        return self.dtask.detect_source(
            gal_array=gal_array,
            fthres=fthres,
            pthres=pthres,
            pratio=pratio,
            bound=bound,
            std_m00=self.std_m00 * self.pix_scale**2.0,
            std_v=self.std_v * self.pix_scale**2.0,
            noise_array=noise_array,
            wdet_cut=wdet_cut,
        )


class FpfsMeasure(FpfsTask):
    """A base class for measurement

    Args:
    psf_array (NDArray): an average PSF image used to initialize the task
    pix_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    nord (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel
    klim_thres (float): the tuncation threshold on Gaussian [default: 1e-20]
    """

    def __init__(
        self,
        psf_array: NDArray,
        pix_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 4,
        klim_thres: float = 1e-20,
    ) -> None:
        super().__init__(
            psf_array=psf_array,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            pix_scale=pix_scale,
            det_nrot=det_nrot,
            klim_thres=klim_thres,
        )
        self.mtask = FpfsImage(
            nx=self.ngrid,
            ny=self.ngrid,
            scale=self.pix_scale,
            sigma_arcsec=self.sigma_arcsec,
            klim=self.klim / self.pix_scale,
            psf_array=psf_array,
            use_estimate=True,
        )
        return

    def run(
        self,
        gal_array: NDArray,
        psf_array: NDArray | None = None,
        det: NDArray | None = None,
        do_rotate: bool = False,
    ) -> NDArray:
        """This function detects galaxy from image

        Args:
        gal_array (NDArray): galaxy image data
        psf_array (NDArray|None): psf image data
        det (list|None): detection catalog
        do_rotate (bool): whether do rotation

        Returns:
            galaxy measurement catalog
        """

        bfunc = np.transpose(self.bfunc, (1, 2, 0))
        return self.mtask.measure_source(
            gal_array=gal_array,
            filter_image=bfunc,
            psf_array=psf_array,
            det=det,
            do_rotate=do_rotate,
        )
