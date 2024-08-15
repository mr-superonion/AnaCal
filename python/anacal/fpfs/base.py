import math

import numpy as np
from numpy.typing import NDArray

from ..base import AnacalBase


def get_shapelets_col_names(norder: int) -> tuple[list, list]:
    """This function returns the column names of shapelet modes
    Args:
    norder (int):  order of the shapelet radial number

    Returns:
    name_s (list): list of the column name
    ind_s (list): list of index in the extended basis function list
    """
    # M_{nm}
    # nm = n*(norder+1)+m
    if norder == 4:
        # This setup is for shear response only
        # Only uses M00, M20, M22 (real and img) and M40, M42 (real and img)
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
        ]
        ind_s = [
            [0, False],
            [10, False],
            [12, False],
            [12, True],
            [20, False],
            [22, False],
            [22, True],
            [24, False],
            [24, True],
        ]
    elif norder == 6:
        # This setup is able to derive kappa response and shear response
        # Only uses M00, M20, M22 (real and img), M40, M42(real and img), M60
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
    else:
        raise ValueError(
            "only support for norder= 4 or norder=6, but your input\
                is norder=%d"
            % norder
        )
    return name_s, ind_s


def get_det_col_names(det_nrot: int) -> list[str]:
    name_d = []
    for irot in range(det_nrot):
        name_d.append("v%d" % irot)
    for irot in range(det_nrot):
        name_d.append("v%dr1" % irot)
    for irot in range(det_nrot):
        name_d.append("v%dr2" % irot)
    return name_d


class ImgBase(AnacalBase):
    """A base class for measurement

    Args:
    pixel_scale (float): pixel scale in arcsec
    sigma_arcsec (float): Shapelet kernel size
    norder (int): the highest order of Shapelets radial components [default: 4]
    det_nrot (int): number of rotation in the detection kernel
    kmax (float | None): maximum k
    psf_array (ndarray): an average PSF image [default: None]
    kmax_thres (float): the tuncation threshold on Gaussian [default: 1e-20]
    mag_zero (float): magnitude zero point [default: 30.0]
    verbose (bool): whether print out INFO
    """

    def __init__(
        self,
        npix: int,
        pixel_scale: float,
        sigma_arcsec: float,
        norder: int = 4,
        det_nrot: int = 4,
        kmax: float | None = None,
        psf_array: NDArray | None = None,
        kmax_thres: float = 1e-20,
        mag_zero: float = 30.0,
        verbose: bool = False,
    ) -> None:
        super().__init__(verbose)
        self.npix = npix
        self.mag_zero = mag_zero
        self.norder = norder
        self.det_nrot = det_nrot
        self.logger.info("Order of the Shapelets: norder=%d" % self.norder)
        self.logger.info(
            "Number of Rotation for detection: det_nrot=%d" % self.det_nrot
        )

        self.sigma_arcsec = sigma_arcsec
        if self.sigma_arcsec > 3.0:
            raise ValueError("sigma_arcsec should be < 3 arcsec")
        if self.norder < 4 and self.det_nrot < 4:
            raise ValueError("Either norder or det_nrot should be >= 4")

        # A few import scales
        self.pixel_scale = pixel_scale
        self._dk = 2.0 * np.pi / self.npix  # assuming pixel scale is 1

        # the following two assumes pixel_scale = 1
        self.sigmaf = float(self.pixel_scale / self.sigma_arcsec)
        self.logger.info(
            "Shapelet kernel in configuration space: sigma= %.4f arcsec"
            % (sigma_arcsec)
        )
        if psf_array is not None:
            if not psf_array.shape == (npix, npix):
                raise ValueError("psf arry has a wrong shape")
        if kmax is None:
            assert psf_array is not None
            # truncation raduis for PSF in Fourier space
            psf_f = np.fft.rfft2(psf_array)
            psf_pow = (np.abs(psf_f) ** 2.0).astype(np.float64)
            self.kmax = (
                get_kmax(
                    psf_pow=psf_pow,
                    sigma=self.sigmaf / np.sqrt(2.0),
                    kmax_thres=kmax_thres,
                )
                * self._dk
            )
        else:
            self.kmax = kmax
        self.logger.info("Maximum |k| for shapelet is %.3f" % (self.kmax))
        return

    def prepare_fpfs_bases(self):
        """This fucntion prepare the FPFS bases (shapelets and detectlets)"""
        bfunc = []
        self.colnames = []
        if self.norder >= 4:
            sfunc, snames = shapelets2d(
                npix=self.npix,
                norder=self.norder,
                sigma=self.sigmaf,
                kmax=self.kmax,
            )
            bfunc.append(sfunc)
            self.colnames = self.colnames + snames
        if self.det_nrot >= 4:
            dfunc, dnames = detlets2d(
                npix=self.npix,
                det_nrot=self.det_nrot,
                sigma=self.sigmaf,
                kmax=self.kmax,
            )
            bfunc.append(dfunc)
            self.colnames = self.colnames + dnames
        self.bfunc = np.vstack(bfunc)
        self.ncol = len(self.colnames)
        self.di = {
            element: index for index, element in enumerate(self.colnames)
        }
        return


def gauss_kernel_rfft(
    ny: int, nx: int, sigma: float, kmax: float, return_grid: bool = False
):
    """Generates a Gaussian kernel on grids for np.fft.rfft transform
    The kernel is truncated at radius kmax.

    Args:
    ny (int): grid size in y-direction
    nx (int): grid size in x-direction
    sigma (float): scale of Gaussian in Fourier space (pixel scale=1)
    kmax (float): upper limit of k
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
    """Generates complex shapelets function in Fourier space, chi00 are
    normalized to 1
    [only support square stamps: ny=nx=npix]

    Args:
    npix (int): number of pixels in x and y direction
    norder (int): radial order of the shaplets
    sigma (float): scale of shapelets in Fourier space
    kmax (float): upper limit of |k|

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


def shapelets2d(npix: int, norder: int, sigma: float, kmax: float):
    """Generates real shapelets function in Fourier space, chi00 are
    normalized to 1
    [only support square stamps: ny=nx=npix]

    Args:
    npix (int): number of pixels in x and y direction
    norder (int): radial order of the shaplets
    sigma (float): scale of shapelets in Fourier space
    kmax (float): upper limit of |k|

    Returns:
    chi_2 (ndarray): 2d shapelet basis w/ shape [n,npix,npix]
    name_s (list): A list of shaplet names
    """
    name_s, ind_s = get_shapelets_col_names(norder)
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
    det_nrot: int,
    sigma: float,
    kmax: float,
):
    """Generates shapelets function in Fourier space, chi00 are normalized to 1
    This function only supports square stamps: ny=nx=npix.

    Args:
    npix (int): number of pixels in x and y direction
    sigma (float): radius of shapelets in Fourier space
    kmax (float): upper limit of |k|
    det_nrot (int): number of rotation in the detection kernel

    Returns:
    psi (ndarray): 2d detlets basis in shape of [det_nrot,3,npix,npix]
    """
    # Gaussian Kernel
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
    name_d = get_det_col_names(det_nrot)
    return psi, name_d


def get_kmax(
    psf_pow: NDArray,
    sigma: float,
    kmax_thres: float = 1e-20,
) -> float:
    """Gets kmax, the region outside kmax is supressed by the shaplet Gaussian
    kernel in FPFS shear estimation method; therefore we set values in this
    region to zeros

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
    """Truncates the input array with square

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
