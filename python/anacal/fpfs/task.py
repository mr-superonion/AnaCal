from ctypes import Array

from . import FpfsImage
from .util import FpfsTask


class FpfsDetect(FpfsTask):
    """A base class for measurement

    Args:
    nx (int):   number of grids in x
    ny (int):   number of grids in y
    psf_array (ndarray):    an average PSF image used to initialize the task
    pix_scale (float):      pixel scale in arcsec
    sigma_arcsec (float):   Shapelet kernel size
    nord (int):             the highest order of Shapelets radial
                            components [default: 4]
    det_nrot (int):         number of rotation in the detection kernel
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        psf_array: Array,
        pix_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 8,
    ) -> None:
        super().__init__(
            psf_array=psf_array,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            pix_scale=pix_scale,
            det_nrot=det_nrot,
        )

        self.dtask = FpfsImage(
            nx=nx,
            ny=ny,
            scale=self.pix_scale,
            sigma_arcsec=self.sigma_arcsec,
            klim=self.klim / self.pix_scale,
            psf_array=self.psf_array,
        )
        self.nx = nx
        self.ny = ny
        return

    def run(
        self,
        gal_array,
        fthres,
        pthres,
        pratio,
        bound,
        std_m00,
        std_v,
        noise_array=None,
    ):
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
        )


class FpfsMeasure(FpfsTask):
    """A base class for measurement

    Args:
    psf_array (ndarray):    an average PSF image used to initialize the task
    pix_scale (float):      pixel scale in arcsec
    sigma_arcsec (float):   Shapelet kernel size
    nord (int):             the highest order of Shapelets radial
                            components [default: 4]
    det_nrot (int):         number of rotation in the detection kernel
    """

    def __init__(
        self,
        psf_array: Array,
        pix_scale: float,
        sigma_arcsec: float,
        nord: int = 4,
        det_nrot: int = 8,
    ) -> None:
        super().__init__(
            psf_array=psf_array,
            sigma_arcsec=sigma_arcsec,
            nord=nord,
            pix_scale=pix_scale,
            det_nrot=det_nrot,
        )
        self.mtask = FpfsImage(
            nx=self.ngrid,
            ny=self.ngrid,
            scale=self.pix_scale,
            sigma_arcsec=self.sigma_arcsec,
            klim=self.klim / self.pix_scale,
            psf_array=self.psf_array,
        )
        return

    def run(
        self,
        gal_array,
        psf_array=None,
        det=None,
    ):
        return self.mtask.measure_source(
            gal_array=gal_array,
            filter_image=self.bfunc,
            psf_array=psf_array,
            det=det,
        )
