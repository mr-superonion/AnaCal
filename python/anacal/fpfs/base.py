import math

import numpy as np
from numpy.typing import NDArray

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
        self.std_r2 = np.sqrt(
            cov_elems[self.di["m00"], self.di["m00"]]
            + cov_elems[self.di["m20"], self.di["m20"]]
            + cov_elems[self.di["m00"], self.di["m20"]]
            + cov_elems[self.di["m20"], self.di["m00"]]
        )
        if self.do_detection:
            self.std_v = np.average(
                np.array(
                    [
                        self.std_modes[self.di["v%d" % _]]
                        for _ in range(det_nrot)
                    ]
                )
            )
        else:
            self.std_v = -1.0
        return cov_elems


def gauss_kernel_rfft(
    ny: int, nx: int, sigma: float, kmax: float, return_grid: bool = False
):
    """Generate a Gaussian kernel on grids for np.fft.rfft transform
    The kernel is truncated at radius kmax. The pixel scale is set to 1

    Args:
    ny (int): grid size in y-direction
    nx (int): grid size in x-direction
    sigma (float): scale of Gaussian in Fourier space (pixel scale=1)
    kmax (float): upper limit of k times scale (klim * scale)
    return_grid (bool): return grids or not

    Returns:
    out (ndarray): Gaussian on grids
    ygrid, xgrid (typle): grids for [y, x] axes, if return_grid
    """
    x = np.fft.rfftfreq(nx, 1 / np.pi / 2.0)
    y = np.fft.fftfreq(ny, 1 / np.pi / 2.0)
    ygrid, xgrid = np.meshgrid(y, x, indexing="ij")
    r2 = xgrid**2.0 + ygrid**2.0
    mask = (r2 <= kmax**2).astype(int)
    out = np.exp(-r2 / 2.0 / sigma**2.0) * mask
    if not return_grid:
        return out
    else:
        return out, (ygrid, xgrid)


def shapelets2d_func(npix: int, norder: int, sigma: float, kmax: float):
    """Generate complex shapelets function in Fourier space, chi00 are
    normalized to 1. The kernel is truncated at radius kmax. The pixel scale is
    set to 1
    [only support square stamps: ny=nx=npix]

    Args:
    npix (int): number of pixels in x and y direction
    norder (int): radial order of the shaplets
    sigma (float): scale of shapelets in Fourier space
    kmax (float): upper limit of k times scale (klim * scale)

    Returns:
    chi (ndarray): 2d shapelet basis
    """

    mord = norder
    gauss_ker, (yfunc, xfunc) = gauss_kernel_rfft(
        npix,
        npix,
        sigma,
        kmax,
        return_grid=True,
    )
    # for inverse Fourier transform
    gauss_ker = gauss_ker / npix**2.0

    rfunc = np.sqrt(xfunc**2.0 + yfunc**2.0)  # radius
    r2_over_sigma2 = (rfunc / sigma) ** 2.0
    ny, nx = gauss_ker.shape

    rmask = rfunc != 0.0
    xtfunc = np.zeros((ny, nx))
    ytfunc = np.zeros((ny, nx))
    np.divide(xfunc, rfunc, where=rmask, out=xtfunc)  # cos(phi)
    np.divide(yfunc, rfunc, where=rmask, out=ytfunc)  # sin(phi)
    eulfunc = xtfunc + 1j * ytfunc  # e^{jphi}
    # Set up Laguerre polynomials
    lfunc = np.zeros((norder + 1, mord + 1, ny, nx), dtype=np.float64)
    lfunc[0, :, :, :] = 1.0
    lfunc[1, :, :, :] = (
        1.0 - r2_over_sigma2 + np.arange(mord + 1)[None, :, None, None]
    )
    #
    chi = np.zeros((norder + 1, mord + 1, ny, nx), dtype=np.complex128)
    for n in range(2, norder + 1):
        for m in range(mord + 1):
            lfunc[n, m, :, :] = (2.0 + (m - 1.0 - r2_over_sigma2) / n) * lfunc[
                n - 1, m, :, :
            ] - (1.0 + (m - 1.0) / n) * lfunc[n - 2, m, :, :]
    for nn in range(norder + 1):
        for mm in range(nn, -1, -2):
            c1 = (nn - abs(mm)) // 2
            d1 = (nn + abs(mm)) // 2
            cc = math.factorial(c1) + 0.0
            dd = math.factorial(d1) + 0.0
            cc = cc / dd
            chi[nn, mm, :, :] = (
                pow(-1.0, d1)
                * pow(cc, 0.5)
                * lfunc[c1, abs(mm), :, :]
                * pow(r2_over_sigma2, abs(mm) / 2)
                * gauss_ker
                * eulfunc**mm
                * (1j) ** nn
            )
    chi = chi.reshape(((norder + 1) ** 2, ny, nx))
    return chi


def shapelets2d(norder: int, npix: int, sigma: float, kmax: float):
    """Generate real shapelets function in Fourier space, chi00 are
    normalized to 1. The kernel is truncated at radius kmax. The pixel scale is
    set to 1
    [only support square stamps: ny=nx=npix]

    Args:
    npix (int): number of pixels in x and y direction
    norder (int): radial order of the shaplets
    sigma (float): scale of shapelets in Fourier space
    kmax (float): upper limit of k times scale (klim * scale)

    Returns:
    chi_2 (ndarray): 2d shapelet basis w/ shape [n,npix,npix]
    name_s (list): A list of shaplet names
    """
    # generate the complex shaplet functions
    chi = shapelets2d_func(npix, norder, sigma, kmax)
    # transform to real shapelet functions
    chi_2 = np.zeros((len(name_s), npix, npix // 2 + 1))
    for i, ind in enumerate(ind_s):
        if ind[1]:
            chi_2[i] = chi[ind[0]].imag
        else:
            chi_2[i] = chi[ind[0]].real
    del chi
    return np.array(chi_2), name_s


def detlets2d(
    npix: int,
    sigma: float,
    kmax: float,
):
    """Generate shapelets function in Fourier space, chi00 are normalized to 1
    This function only supports square stamps: ny=nx=npix. The kernel is
    truncated at radius kmax. The pixel scale is set to 1

    Args:
    npix (int): number of pixels in x and y direction
    sigma (float): radius of shapelets in Fourier space
    kmax (float): upper limit of k times scale (klim * scale)

    Returns:
    psi (ndarray): 2d detlets basis in shape of [det_nrot,3,npix,npix]
    """
    # Gaussian kernel
    gauss_ker, (k2grid, k1grid) = gauss_kernel_rfft(
        npix,
        npix,
        sigma,
        kmax,
        return_grid=True,
    )
    # for inverse Fourier transform
    gauss_ker = gauss_ker / npix**2.0

    # for shear response
    q1_ker = (k1grid**2.0 - k2grid**2.0) / sigma**2.0 * gauss_ker
    q2_ker = (2.0 * k1grid * k2grid) / sigma**2.0 * gauss_ker
    # quantities for neighbouring pixels
    d1_ker = (-1j * k1grid) * gauss_ker
    d2_ker = (-1j * k2grid) * gauss_ker
    # initial output psi function
    ny, nx = gauss_ker.shape
    psi = np.zeros((3, det_nrot, ny, nx), dtype=np.complex128)
    for irot in range(det_nrot):
        x = np.cos(2.0 * np.pi / det_nrot * irot)
        y = np.sin(2.0 * np.pi / det_nrot * irot)
        foub = np.exp(1j * (k1grid * x + k2grid * y))
        psi[0, irot] = gauss_ker - gauss_ker * foub
        psi[1, irot] = q1_ker - (q1_ker + x * d1_ker - y * d2_ker) * foub
        psi[2, irot] = q2_ker - (q2_ker + y * d1_ker + x * d2_ker) * foub
    psi = psi.reshape(3 * det_nrot, ny, nx)
    return psi, name_d


def get_kmax(
    psf_pow: NDArray,
    sigma: float,
    kmax_thres: float = 1e-20,
) -> float:
    """Measure kmax, the region outside kmax is supressed by the shaplet
    Gaussian kernel in FPFS shear estimation method; therefore we set values in
    this region to zeros

    Args:
    psf_pow (ndarray): PSF's Fourier power (rfft)
    sigma (float): one sigma of Gaussian Fourier power (pixel scale=1)
    kmax_thres (float): the tuncation threshold on Gaussian [default: 1e-20]

    Returns:
    kmax (float): the limit radius
    """
    npix = psf_pow.shape[0]
    gaussian, (y, x) = gauss_kernel_rfft(
        npix,
        npix,
        sigma,
        np.pi,
        return_grid=True,
    )
    r = np.sqrt(x**2.0 + y**2.0)  # radius
    mask = gaussian / psf_pow < kmax_thres
    dk = 2.0 * math.pi / npix
    kmax_pix = round(float(np.min(r[mask]) / dk))
    kmax_pix = min(max(kmax_pix, npix // 5), npix // 2 - 1)
    return kmax_pix


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
