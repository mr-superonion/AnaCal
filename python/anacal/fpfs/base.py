import numpy as np
from numpy.typing import NDArray

from .._anacal.fpfs import (
    detlets2d as _detlets2d,
    gauss_kernel_rfft as _gauss_kernel_rfft,
    get_kmax as _get_kmax,
    shapelets2d as _shapelets2d,
    shapelets2d_func as _shapelets2d_func,
)
from ..base import AnacalBase
from . import Image

# M_{nm}
# nm = n*(norder+1)+m
# This setup is able to derive kappa response and shear response
# Only uses M00, M20, M22 (real and img), M40, M42(real and img), M60
norder_shapelets = 6
name_s = [
    "m00",
    "m20",
    "m22c",
    "m22s",
    "m40",
    "m42c",
    "m42s",
    "m44c",
    "m44s",
    "m60",
    "m64c",
    "m64s",
]
ind_s = [
    [0, False],
    [14, False],
    [16, False],
    [16, True],
    [28, False],
    [30, False],
    [30, True],
    [32, False],
    [32, True],
    [42, False],
    [46, False],
    [46, True],
]

det_nrot = 4
name_d = [
    "v0",
    "v1",
    "v2",
    "v3",
    "v0r1",
    "v1r1",
    "v2r1",
    "v3r1",
    "v0r2",
    "v1r2",
    "v2r2",
    "v3r2",
]


class FpfsKernel(AnacalBase):
    """Fpfs measurement kernel object

    Args:
    npix (int): number of pixels in a postage stamp
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    kmax (float | None): maximum k
    psf_array (ndarray): an average PSF image [default: None]
    kmax_thres (float): the tuncation threshold on Gaussian [default: 1e-20]
    do_detection (bool): whether compute detection kernel
    """

    def __init__(
        self,
        *,
        npix: int,
        pixel_scale: float,
        sigma_arcsec: float,
        kmax: float | None = None,
        psf_array: NDArray | None = None,
        kmax_thres: float = 1e-20,
        do_detection: bool = True,
        verbose=False,
    ) -> None:
        super().__init__(verbose=verbose)
        self.npix = npix
        self.do_detection = do_detection

        self.sigma_arcsec = sigma_arcsec
        if self.sigma_arcsec > 3.0:
            raise ValueError("sigma_arcsec should be < 3 arcsec")

        # A few import scales
        self.pixel_scale = pixel_scale
        self._dk = 2.0 * np.pi / self.npix  # assuming pixel scale is 1

        # the following two assumes pixel_scale = 1
        self.sigmaf = float(self.pixel_scale / self.sigma_arcsec)
        if psf_array is None:
            # make a delta psf
            psf_array = np.zeros((npix, npix))
            psf_array[npix // 2, npix // 2] = 1
        else:
            if not psf_array.shape == (npix, npix):
                raise ValueError("psf arry has a wrong shape")

        psf_f = np.fft.rfft2(psf_array)
        self.psf_array = psf_array
        self.psf_pow = (np.abs(psf_f) ** 2.0).astype(np.float64)
        if kmax is None:
            assert psf_array is not None
            # truncation raduis for PSF in Fourier space
            self.kmax = (
                get_kmax(
                    psf_pow=self.psf_pow,
                    sigma=self.sigmaf / np.sqrt(2.0),
                    kmax_thres=kmax_thres,
                )
                * self._dk
            )
        else:
            self.kmax = kmax

        self.prepare_fpfs_bases()
        return

    def prepare_fpfs_bases(self):
        """This fucntion prepare the FPFS bases (shapelets and detectlets)"""
        bfunc = []
        self.colnames = []
        sfunc, snames = shapelets2d(
            norder=norder_shapelets,
            npix=self.npix,
            sigma=self.sigmaf,
            kmax=self.kmax,
        )
        bfunc.append(sfunc)
        self.colnames = self.colnames + snames
        if self.do_detection:
            dfunc, dnames = detlets2d(
                npix=self.npix,
                sigma=self.sigmaf,
                kmax=self.kmax,
            )
            bfunc.append(dfunc)
            self.colnames = self.colnames + dnames
        self.bfunc = np.vstack(bfunc)
        self.ncol = len(self.colnames)
        self.dtype = [(name, "f8") for name in self.colnames]
        self.bfunc_use = np.transpose(self.bfunc, (1, 2, 0))
        self.di = {
            element: index for index, element in enumerate(self.colnames)
        }
        return

    def prepare_covariance(
        self, variance: float, noise_pf: NDArray | None = None
    ):
        """Estimate covariance of measurement error

        Args:
        variance (float): Noise variance
        noise_pf (NDArray | None): Power spectrum (assuming homogeneous) of
        noise
        """
        variance = variance * 2.0
        if noise_pf is not None:
            if noise_pf.shape == (self.npix, self.npix // 2 + 1):
                # rfft
                noise_pf = np.array(noise_pf, dtype=np.float64)
            elif noise_pf.shape == (self.npix, self.npix):
                # fft
                noise_pf = np.fft.ifftshift(noise_pf)
                noise_pf = np.array(
                    noise_pf[:, : self.npix // 2 + 1], dtype=np.float64
                )
            else:
                raise ValueError("noise power not in correct shape")
        else:
            ss = (self.npix, self.npix // 2 + 1)
            noise_pf = np.ones(ss)
        norm_factor = variance * self.npix**2.0 / noise_pf[0, 0]
        noise_pf = noise_pf * norm_factor

        img_obj = Image(nx=self.npix, ny=self.npix, scale=self.pixel_scale)
        img_obj.set_f(noise_pf)
        img_obj.deconvolve(
            psf_image=self.psf_pow,
            klim=self.kmax / self.pixel_scale,
        )
        noise_pf_deconv = img_obj.draw_f().real
        del img_obj

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
        self.std_modes = np.sqrt(np.diagonal(cov_elems))
        self.std_m00 = self.std_modes[self.di["m00"]]
        return cov_elems


def gauss_kernel_rfft(
    ny: int, nx: int, sigma: float, kmax: float, return_grid: bool = False
):
    """Generate a Gaussian kernel on grids for :func:`numpy.fft.rfft`.

    This function is provided for backwards compatibility with the original
    NumPy implementation but now delegates to the high-performance C++
    extension.
    """

    return _gauss_kernel_rfft(ny, nx, sigma, kmax, return_grid)


def shapelets2d_func(npix: int, norder: int, sigma: float, kmax: float):
    """Generate complex shapelet functions in Fourier space."""

    return _shapelets2d_func(npix, norder, sigma, kmax)


def shapelets2d(norder: int, npix: int, sigma: float, kmax: float):
    """Generate real-valued shapelet functions in Fourier space."""

    chi = _shapelets2d(norder, npix, sigma, kmax)
    return np.array(chi), name_s


def detlets2d(
    npix: int,
    sigma: float,
    kmax: float,
):
    """Generate complex detection basis functions in Fourier space."""

    psi = _detlets2d(npix, sigma, kmax)
    return psi, name_d


def get_kmax(
    psf_pow: NDArray,
    sigma: float,
    kmax_thres: float = 1e-20,
) -> float:
    """Estimate the truncation radius ``kmax`` for the Gaussian kernel."""

    return _get_kmax(psf_pow, sigma, kmax_thres)


def truncate_square(arr: NDArray, rcut: int) -> None:
    """Truncate the input array with square

    Args:
    arr (ndarray): image array
    rcut (int): radius of the square (width / 2)
    """
    if len(arr.shape) != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Input array must be a 2D square array")
    npix = arr.shape[0]
    npix2 = npix // 2
    assert rcut < npix2, "truncation radius too large."
    if rcut < npix2 - 1:
        arr[: npix2 - rcut, :] = 0
        arr[npix2 + rcut + 1 :, :] = 0
        arr[:, : npix2 - rcut] = 0
        arr[:, npix2 + rcut + 1 :] = 0
    return
