import numpy as np
from numpy.typing import NDArray

from . import FpfsImage
from .util import FpfsTask


class FpfsDetect(FpfsTask):
    """A base class for measurement

    Args:
    nx (int): number of grids in x
    ny (int): number of grids in y
    psf_array (ndarray): an average PSF image used to initialize the task
    pix_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    nord (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel [default: 8]
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        psf_array: NDArray,
        pix_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 8,
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
            psf_array=self.psf_array,
            use_estimate=True,
        )
        self.nx = nx
        self.ny = ny
        return

    def run(
        self,
        gal_array: NDArray,
        fthres: float,
        pthres: float,
        pratio: float,
        bound: int,
        std_m00: float,
        std_v: float,
        noise_array: NDArray | None = None,
        wdet_cut: float = 0.0,
    ) -> NDArray:
        """This function detects galaxy from image

        Args:
        gal_array (ndarray): galaxy image data
        fthres (float): flux threshold
        pthres (float): peak threshold
        pratio (float): peak flux ratio
        bound (int): minimum distance to boundary
        std_m00 (float): standard deviation of m00 measurement error
        std_v (float): standard deviation of v measurement error
        noise_array (ndarray|None): noise array

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
            std_m00=std_m00 * self.pix_scale**2.0,
            std_v=std_v * self.pix_scale**2.0,
            noise_array=noise_array,
            wdet_cut=wdet_cut,
        )


class FpfsMeasure(FpfsTask):
    """A base class for measurement

    Args:
    psf_array (ndarray): an average PSF image used to initialize the task
    pix_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    nord (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel
    """

    def __init__(
        self,
        psf_array: NDArray,
        pix_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 8,
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
            psf_array=self.psf_array,
            use_estimate=True,
        )
        return

    def run(
        self,
        gal_array: NDArray,
        psf_array=None,
        det=None,
        do_rotate=False,
    ) -> NDArray:
        """This function detects galaxy from image

        Args:
        gal_array (ndarray): galaxy image data
        psf_array (ndarray|None): psf image data
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
