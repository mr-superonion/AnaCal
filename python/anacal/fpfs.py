from dataclasses import dataclass

import numpy as np
import numpy.lib.recfunctions as rfn
from numpy.typing import NDArray

from ._anacal.fpfs import (
    FpfsImage,
)
from ._anacal.fpfs import detlets2d as _detlets2d
from ._anacal.fpfs import (
    fpfs_cut_sigma_ratio,
    fpfs_det_sigma2,
)
from ._anacal.fpfs import gauss_kernel_rfft as _gauss_kernel_rfft
from ._anacal.fpfs import get_kmax as _get_kmax
from ._anacal.fpfs import m00_to_flux as _m00_to_flux
from ._anacal.fpfs import (
    measure_fpfs,
    measure_fpfs_shape,
    measure_fpfs_wdet,
    measure_fpfs_wdet0,
    measure_fpfs_wsel,
    measure_shapelets_dg,
)
from ._anacal.fpfs import shapelets2d as _shapelets2d
from ._anacal.fpfs import shapelets2d_func as _shapelets2d_func
from ._anacal.image import Image
from .psf import BasePsf

npix_patch = 256
npix_overlap = 64


norder_shapelets = 6


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
    """Generate complex shapelet basis functions in Fourier space.

    Args:
        npix (int): Number of pixels along each axis of the square grid.
        norder (int): Maximum shapelet order.
        sigma (float): Gaussian kernel scale in Fourier-space units
            (``pixel_scale / sigma_arcsec``).
        kmax (float): Truncation radius in Fourier space.

    Returns:
        NDArray: Complex shapelet basis array.
    """

    return _shapelets2d_func(npix, norder, sigma, kmax)


def shapelets2d(norder: int, npix: int, sigma: float, kmax: float):
    """Generate real-valued shapelet basis functions in Fourier space.

    The returned basis contains the modes needed to derive convergence and
    shear responses: M00, M20, M22 (real & imag), M40, M42 (real & imag),
    M44 (real & imag), M60, M64c, M64s.

    Args:
        norder (int): Maximum shapelet order.  Must be 2, 4, or 6.
        npix (int): Number of pixels along each axis of the square grid.
        sigma (float): Gaussian kernel scale in Fourier-space units.
        kmax (float): Truncation radius in Fourier space.

    Returns:
        tuple[NDArray, list[str]]: A tuple of ``(basis_functions, names)``
        where *basis_functions* has shape ``(nmodes, npix, npix//2+1)``
        and *names* lists the corresponding mode labels.
    """
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
    chi = _shapelets2d(npix, sigma, kmax)
    if norder == 6:
        return np.array(chi), name_s
    elif norder == 4:
        return np.array(chi)[0:9], name_s[0:9]
    elif norder == 2:
        return np.array(chi)[0:4], name_s[0:4]


def detlets2d(
    npix: int,
    sigma: float,
    kmax: float,
):
    """Generate detection basis functions (detlets) in Fourier space.

    Detlets are used for peak detection and their shear derivatives.

    Args:
        npix (int): Number of pixels along each axis of the square grid.
        sigma (float): Gaussian kernel scale in Fourier-space units.
        kmax (float): Truncation radius in Fourier space.

    Returns:
        tuple[NDArray, list[str]]: A tuple of ``(basis_functions, names)``
        where *basis_functions* has shape ``(12, npix, npix//2+1)``
        and *names* lists the detection mode labels (v0–v3 and their
        shear derivatives).
    """
    name_d = [
        "v0",
        "v1",
        "v2",
        "v3",
        "dv0_dg1",
        "dv1_dg1",
        "dv2_dg1",
        "dv3_dg1",
        "dv0_dg2",
        "dv1_dg2",
        "dv2_dg2",
        "dv3_dg2",
    ]
    psi = _detlets2d(npix, sigma, kmax)
    return psi, name_d


def get_kmax(
    psf_pow: NDArray,
    sigma: float,
    kmax_thres: float = 1e-20,
) -> float:
    """Estimate the truncation radius ``kmax`` for the Gaussian kernel.

    Finds the largest wavenumber at which the product of the Gaussian
    kernel and the PSF power spectrum exceeds *kmax_thres*.

    Args:
        psf_pow (NDArray): PSF power spectrum ``|FFT(psf)|^2``, shape
            ``(npix, npix // 2 + 1)``.
        sigma (float): Gaussian kernel scale in Fourier-space units.
        kmax_thres (float): Threshold below which the kernel is truncated.

    Returns:
        float: Truncation wavenumber in grid units.
    """
    return _get_kmax(psf_pow, sigma, kmax_thres)


def m00_to_flux(
    m00: float | NDArray,
    sigma_shapelets: float,
):
    """Convert the ``m00`` shapelet coefficient to flux.

    Parameters
    ----------
    m00
        Scalar or array of monopole shapelet coefficients.
    sigma_shapelets
        sigma of Gaussian kernel for shapelets, in arcseconds.

    Returns
    -------
    float | NDArray
        Flux values corresponding to the provided ``m00`` coefficients.
        Equal to ``m00 * 2 * pi * sigma_shapelets**2`` (the original
        ``pixel_scale`` factor cancels and is no longer needed).
    """

    return _m00_to_flux(m00, sigma_shapelets)


def _append_flux_gauss(meas, sigma_shapelets, std_m00, kernel):
    """Append Gaussian-aperture flux columns and finalize per-kernel names.

    Adds seven columns to a per-kernel ``measure_fpfs`` output:

      - ``flux``       = m00_to_flux(``m00``)
      - ``dflux_dg1``  = m00_to_flux(``dm00_dg1``)
      - ``dflux_dg2``  = m00_to_flux(``dm00_dg2``)
      - ``flux_err``   = m00_to_flux(``std_m00``)  (per-kernel
        constant broadcast to N rows)
      - ``s2n``        = ``flux`` / ``flux_err``
      - ``ds2n_dg1``   = ``dflux_dg1`` / ``flux_err``
      - ``ds2n_dg2``   = ``dflux_dg2`` / ``flux_err``

    The constant 2*pi*sigma_shapelets^2 is the same conversion used in
    ``m00_to_flux`` and cancels in flux/err ratios.

    Then names every column for ``kernel`` via :func:`_kernel_rename_map`
    (flux family in photoz token order ``flux_{kernel}`` etc.; shapelet
    moments keep the ``{kernel}_`` prefix), so the caller needs no further
    rename.
    """
    n = len(meas)
    m00 = np.ascontiguousarray(meas["m00"], dtype=np.float64)
    dm00_dg1 = np.ascontiguousarray(meas["dm00_dg1"], dtype=np.float64)
    dm00_dg2 = np.ascontiguousarray(meas["dm00_dg2"], dtype=np.float64)
    flux = _m00_to_flux(m00, sigma_shapelets)
    dflux_dg1 = _m00_to_flux(dm00_dg1, sigma_shapelets)
    dflux_dg2 = _m00_to_flux(dm00_dg2, sigma_shapelets)
    flux_err = _m00_to_flux(float(std_m00), sigma_shapelets)
    # signal-to-noise and its shear response (flux_err is a shear-independent
    # per-kernel noise constant, so ds2n_dg = dflux_dg / flux_err)
    s2n = flux / flux_err
    ds2n_dg1 = dflux_dg1 / flux_err
    ds2n_dg2 = dflux_dg2 / flux_err
    meas = rfn.append_fields(
        meas,
        ["flux", "dflux_dg1", "dflux_dg2", "flux_err",
         "s2n", "ds2n_dg1", "ds2n_dg2"],
        [flux, dflux_dg1, dflux_dg2,
         np.full(n, flux_err, dtype=np.float64),
         s2n, ds2n_dg1, ds2n_dg2],
        usemask=False,
    )
    return rfn.rename_fields(meas, _kernel_rename_map(meas.dtype.names, kernel))


# Flux-family columns (appended by _append_flux_gauss) carry the kernel token
# AFTER the observable — the "photoz" order consumed by photoZPipe and the
# xlens merge (flux_{k}, dflux_{k}_dg1, flux_{k}_err, s2n_{k}, ...). Every other
# column (the shapelet moments) keeps the {kernel}_{name} prefix.
_FLUX_PHOTOZ_TEMPLATE = {
    "flux": "flux_{k}",
    "flux_err": "flux_{k}_err",
    "dflux_dg1": "dflux_{k}_dg1",
    "dflux_dg2": "dflux_{k}_dg2",
    "s2n": "s2n_{k}",
    "ds2n_dg1": "ds2n_{k}_dg1",
    "ds2n_dg2": "ds2n_{k}_dg2",
}


def _kernel_rename_map(names, kernel):
    """Build the per-kernel field-rename map.

    Flux-family columns take the photoz token order (``flux_{kernel}``,
    ``dflux_{kernel}_dg1``, ``flux_{kernel}_err``, ``s2n_{kernel}``, ...); every
    other column keeps the ``{kernel}_{name}`` prefix.
    """
    out = {}
    for name in names:
        tmpl = _FLUX_PHOTOZ_TEMPLATE.get(name)
        out[name] = tmpl.format(k=kernel) if tmpl else f"{kernel}_{name}"
    return out


class FpfsTask:
    """FPFS measurement task for shapelet-based galaxy shape measurement.

    Prepares shapelet and detection basis functions, then measures galaxy
    shapes at detected positions via PSF-deconvolved Fourier-space filtering.

    Args:
        npix (int): Number of pixels per side of a postage stamp.
        pixel_scale (float): Pixel scale in arcseconds.
        sigma_shapelets (float): Shapelet Gaussian kernel size in arcseconds.
            Must be less than 3.0.
        noise_variance (float): Variance of the image noise per pixel.
            Must be positive when *do_detection* is ``True``.
        kmax (float | None): Maximum wavenumber for Fourier-space truncation.
            If ``None``, estimated automatically from the PSF.
        psf_array (NDArray | None): Average PSF image of shape
            ``(npix, npix)``.  Defaults to a delta function at centre.
        kmax_thres (float): Threshold for automatic ``kmax`` estimation.
        do_detection (bool): Whether to prepare detection kernels and
            compute noise covariance.
        bound (int): Minimum distance to the image boundary for source
            detection.
    """

    def __init__(
        self,
        *,
        npix: int,
        pixel_scale: float,
        sigma_shapelets: float,
        noise_variance: float = -1,
        kmax: float | None = None,
        psf_array: NDArray | None = None,
        kmax_thres: float = 1e-20,
        do_detection: bool = True,
        bound: int = 0,
    ) -> None:
        self.npix = npix
        self.do_detection = do_detection

        self.sigma_shapelets = sigma_shapelets
        if self.sigma_shapelets > 3.0:
            raise ValueError("sigma_shapelets should be < 3 arcsec")

        self.pixel_scale = pixel_scale
        self._dk = 2.0 * np.pi / self.npix

        self.sigmaf = float(self.pixel_scale / self.sigma_shapelets)
        if psf_array is None:
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
        klim = 1e10 if self.kmax is None else (self.kmax / self.pixel_scale)
        self.mtask = FpfsImage(
            nx=self.npix,
            ny=self.npix,
            scale=self.pixel_scale,
            sigma_arcsec=self.sigma_shapelets,
            klim=klim,
            psf_array=self.psf_array,
            use_estimate=True,
        )

        if self.do_detection:
            self.dtask = FpfsImage(
                nx=npix_patch,
                ny=npix_patch,
                scale=self.pixel_scale,
                sigma_arcsec=self.sigma_shapelets,
                klim=klim,
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

    def prepare_fpfs_bases(self):
        """Prepare the FPFS bases (shapelets and detectlets)."""

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

    def calculate_covariance(
        self,
        variance: float,
        psf_pow: NDArray,
        noise_pf: NDArray | None = None,
    ):
        """Calculate covariance of measurement error for a single PSF.

        This computes the covariance matrix of FPFS measurement modes
        due to image noise, for a given PSF power spectrum. Unlike
        :meth:`prepare_covariance`, this does **not** double the variance.

        Args:
            variance (float): noise variance per pixel
            psf_pow (NDArray): PSF power spectrum |FFT(psf)|^2,
                shape ``(npix, npix // 2 + 1)``
            noise_pf (NDArray | None): noise power spectrum [default: None]

        Returns:
            NDArray: covariance matrix of shape ``(ncol, ncol)``
        """
        if noise_pf is not None:
            if noise_pf.shape == (self.npix, self.npix // 2 + 1):
                noise_pf = np.array(noise_pf, dtype=np.float64)
            elif noise_pf.shape == (self.npix, self.npix):
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
            psf_image=psf_pow,
            klim=self.kmax / self.pixel_scale,
        )
        noise_pf_deconv = img_obj.draw_f().real
        del img_obj

        _w = np.ones(psf_pow.shape) * 2.0
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
        return cov_elems

    def _rotate_rfft_pow(self, pf: NDArray) -> NDArray:
        """Rotate a power spectrum in rfft format 90 degrees CW around DC.

        CW 90-degree rotation: ``P_rot[j, i] = P[(N-i)%N, j]``
        in unshifted FFT format.

        Args:
            pf (NDArray): power spectrum in rfft format ``(npix, npix//2+1)``

        Returns:
            NDArray: rotated power spectrum in rfft format.
        """
        n = self.npix
        nhalf = n // 2 + 1
        # Reconstruct full (n, n) spectrum using Hermitian symmetry
        full = np.empty((n, n), dtype=np.float64)
        full[:, :nhalf] = pf
        full[0, nhalf:] = pf[0, nhalf - 2 : 0 : -1]
        full[1:, nhalf:] = pf[-1:0:-1, nhalf - 2 : 0 : -1]
        # CW 90-degree rotation around DC at (0, 0)
        col_src = (n - np.arange(n)) % n
        rotated = full[col_src, :].T
        return rotated[:, :nhalf].astype(np.float64).copy()

    def _rotate_noise_pf(self, noise_pf: NDArray) -> NDArray:
        """Rotate noise power spectrum 90 degrees CW in Fourier space.

        Accepts noise_pf in either full fftshifted ``(npix, npix)`` or
        rfft ``(npix, npix//2+1)`` format, converts to rfft, and delegates
        to :meth:`_rotate_rfft_pow`.

        Args:
            noise_pf (NDArray): noise power spectrum.

        Returns:
            NDArray: rotated noise power spectrum in rfft format.
        """
        n = self.npix
        nhalf = n // 2 + 1
        if noise_pf.shape == (n, nhalf):
            pf = np.array(noise_pf, dtype=np.float64)
        elif noise_pf.shape == (n, n):
            pf = np.fft.ifftshift(noise_pf)
            pf = np.array(pf[:, :nhalf], dtype=np.float64)
        else:
            raise ValueError("noise power not in correct shape")
        return self._rotate_rfft_pow(pf)

    def prepare_covariance(
        self, variance: float, noise_pf: NDArray | None = None
    ):
        """Estimate covariance of measurement error.

        The total covariance accounts for noise from both the galaxy image
        measurement and the rotated noise image used for noise bias
        subtraction. It is the sum of :meth:`calculate_covariance` called
        with the original PSF and with the PSF rotated 90 degrees CW
        in Fourier space (matching the C++ ``rotate90_f``).

        Args:
            variance (float): noise variance per pixel
            noise_pf (NDArray | None): noise power spectrum [default: None]

        Returns:
            NDArray: covariance matrix of shape ``(ncol, ncol)``
        """
        # Covariance from the galaxy image measurement (original PSF)
        cov_elems = self.calculate_covariance(
            variance=variance,
            psf_pow=self.psf_pow,
            noise_pf=noise_pf,
        )

        # Covariance from the rotated noise measurement.
        # The C++ code rotates the PSF in Fourier space (rotate90_f)
        # before deconvolution. The noise power spectrum is also rotated
        # (matching xlens's rotate_noise_corr for correlated noise).
        # Both use CW 90-degree rotation around DC at (0, 0).
        psf_rot_pow = self._rotate_rfft_pow(self.psf_pow)
        if noise_pf is not None:
            noise_pf_rot = self._rotate_noise_pf(noise_pf)
        else:
            noise_pf_rot = None
        cov_elems = cov_elems + self.calculate_covariance(
            variance=variance,
            psf_pow=psf_rot_pow,
            noise_pf=noise_pf_rot,
        )

        self.std_modes = np.sqrt(np.diagonal(cov_elems))
        self.std_m00 = self.std_modes[self.di["m00"]]
        return cov_elems

    def detect(
        self,
        *,
        gal_array: NDArray,
        fthres: float,
        pthres: float,
        omega_v: float,
        v_min: float,
        noise_array: NDArray | None = None,
        mask_array: NDArray | None = None,
    ) -> NDArray:
        """Detect galaxies in an image using FPFS detection modes.

        Args:
            gal_array (NDArray): Galaxy image array.
            fthres (float): Flux signal-to-noise threshold for detection.
            pthres (float): Peak identification threshold for pooling.
            omega_v (float): Smoothness parameter for the v-mode cut.
            v_min (float): Minimum value of the v detection mode.
            noise_array (NDArray | None): Pure noise image for noise-bias
                subtraction.
            mask_array (NDArray | None): Mask image (1 for masked pixels).

        Returns:
            NDArray: Structured array with columns ``('y', 'x')`` giving
            detected source positions.
        """
        assert self.dtask is not None
        return self.dtask.detect_source(
            gal_array=gal_array,
            fthres=fthres,
            pthres=pthres,
            std_m00=self.std_m00 * self.pixel_scale**2.0,
            omega_v=omega_v * self.pixel_scale**2.0,
            v_min=v_min * self.pixel_scale**2.0,
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
        """Measure galaxy shapes using a spatially constant PSF image.

        Args:
            gal_array (NDArray): Galaxy image array.
            psf_array (NDArray): PSF image of shape ``(npix, npix)``.
            det (NDArray | None): Detection catalog with ``('y', 'x')``
                columns.
            noise_array (NDArray | None): Pure noise image for noise-bias
                subtraction.

        Returns:
            tuple[NDArray, NDArray | None]: ``(src_g, src_n)`` where
            *src_g* is the source measurement array (noise-bias corrected
            when *noise_array* is given) and *src_n* is the noise
            measurement array (or ``None``).
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
        """Measure galaxy shapes using a spatially varying PSF model.

        Evaluates the PSF at each detected position via *psf_obj.draw()*
        and measures shapelet modes per object.

        Args:
            gal_array (NDArray): Galaxy image array.
            psf_obj (BasePsf): Spatially varying PSF model.
            det (NDArray): Detection catalog with ``('y', 'x')`` columns.
            noise_array (NDArray | None): Pure noise image for noise-bias
                subtraction.

        Returns:
            tuple[NDArray, NDArray | None]: ``(src_g, src_n)`` — see
            :meth:`run_psf_array`.
        """
        # self.logger.warning("Input PSF is python object")
        src_g = []
        src_n = []
        for _d in det:
            this_psf_array = psf_obj.draw(x=_d["x"], y=_d["y"])
            srow = self.mtask.measure_source_at(
                gal_array=gal_array,
                filter_image=self.bfunc_use,
                psf_array=this_psf_array,
                y=_d["y"],
                x=_d["x"],
                do_rotate=False,
            )
            if noise_array is not None:
                nrow = self.mtask.measure_source_at(
                    gal_array=noise_array,
                    filter_image=self.bfunc_use,
                    psf_array=this_psf_array,
                    y=_d["y"],
                    x=_d["x"],
                    do_rotate=True,
                )
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


    def run(
        self,
        gal_array: NDArray,
        psf: BasePsf | NDArray,
        det: NDArray | None = None,
        noise_array: NDArray | None = None,
    ):
        """Measure FPFS shapelet modes at detected positions.

        Dispatches to :meth:`run_psf_array` or :meth:`run_psf_python`
        depending on whether *psf* is an array or a :class:`BasePsf`
        object.

        Args:
            gal_array (NDArray): Galaxy image array.
            psf (BasePsf | NDArray): PSF image ``(npix, npix)`` or a
                spatially varying :class:`BasePsf` model.
            det (NDArray | None): Detection catalog with ``('y', 'x')``
                columns.  Required when *psf* is a :class:`BasePsf`.
            noise_array (NDArray | None): Pure noise image for noise-bias
                subtraction.

        Returns:
            dict: ``{"data": src_g, "noise": src_n}`` where *src_g* is
            a structured array of FPFS mode measurements and *src_n* is
            the corresponding noise measurements (or ``None``).
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


@dataclass
class FpfsConfig:
    """Configuration parameters for the FPFS measurement pipeline.

    All threshold and smoothness parameters are specified in units that
    assume a magnitude zero-point of 30.  They are rescaled internally
    by :func:`process_image` to match the actual zero-point.
    """

    npix: int = 64
    """size of the stamp before Fourier Transform"""

    kmax_thres: float = 1e-12
    """The threshold used to define the upper limit of k we use in Fourier
    space."""

    bound: int = 35
    """Boundary buffer length, the sources in the buffer reion are not
    counted."""

    sigma_shapelets: float = 0.52
    """Smoothing scale of the shapelet and detection kernel."""

    sigma_shapelets1: float = -1
    """Smoothing scale of the second shapelet kernel."""

    sigma_shapelets2: float = -1
    """Smoothing scale of the third shapelet kernel."""

    pthres: float = 0.12
    """Detection threshold (peak identification) for the pooling."""

    fthres: float = 8.0
    """Detection threshold (minimum signal-to-noise ratio) for the first
    pooling."""

    omega_r2: float = 4.8
    """smoothness parameter for r2 cut"""

    r2_min: float = 0.1
    """Minimum trace moment matrix"""

    omega_v: float = 0.9
    """smoothness parameter for v cut"""

    v_min: float = 0.45
    """Minimum of v"""

    snr_min: float = 12
    """Minimum Signal-to-Noise Ratio for detection."""

    c0: float = 8.4
    """Weighting parameter for m00 for ellipticity definition."""


def _rename_linear_fields(
    arr: np.ndarray,
    *,
    prefix: str,
    is_noise: bool,
    base_column_name: str | None,
) -> np.ndarray | None:
    """Rename structured-array fields for linear modes.

    - data*:   <name> -> f"{prefix}{name}"
    - noise*:  mXX... -> nXX..., then -> f"{prefix}{nXX...}"
    """
    if arr is None:
        return arr
    if arr.dtype.names is None:
        raise TypeError("Expected a structured array (names is None).")

    mapping: dict[str, str] = {}
    for name in arr.dtype.names:
        new = name
        if is_noise:
            if name.startswith("m"):
                new = "n" + name[1:]
            elif name.startswith("dv"):
                new = "du" + name[2:]
            elif name.startswith("v"):
                new = "u" + name[1:]
        if base_column_name is None:
            mapping[name] = prefix + new
        else:
            mapping[name] = base_column_name + prefix + new
    return rfn.rename_fields(arr, mapping)


def _pack_linear_modes(linear_modes, base_column_name):
    """Merge all linear-mode arrays into a single structured array.

    Args:
        linear_modes (dict): Mapping from mode keys (``"detection"``,
            ``"data"``, ``"noise"``, ``"data1"``, etc.) to arrays.
        base_column_name (str | None): Optional prefix for column names.

    Returns:
        NDArray: Merged structured array.
    """
    blocks: list[np.ndarray] = []
    if "detection" in linear_modes and linear_modes["detection"] is not None:
        blocks.append(linear_modes["detection"])
    for tag in ["", "1", "2"]:
        if f"data{tag}" in linear_modes:
            blocks.append(
                _rename_linear_fields(
                    linear_modes[f"data{tag}"],
                    prefix=f"fpfs{tag}_",
                    is_noise=False,
                    base_column_name=base_column_name,
                )
            )
        if f"noise{tag}" in linear_modes:
            if linear_modes[f"noise{tag}"] is None:
                noise = np.zeros(
                    len(linear_modes[f"data{tag}"]),
                    dtype=linear_modes[f"data{tag}"].dtype,
                )
            else:
                noise = linear_modes[f"noise{tag}"]
            blocks.append(
                _rename_linear_fields(
                    noise,
                    prefix=f"fpfs{tag}_",
                    is_noise=True,
                    base_column_name=base_column_name,
                )
            )
        # Per-band per-kernel m00 standard deviation.  Stored as a
        # broadcast scalar so the column lines up with all the per-mode
        # columns above.
        std_key = "std_m00" if tag == "" else f"std_m00_{tag}"
        if std_key in linear_modes:
            n_rows = len(linear_modes[f"data{tag}"])
            err_name = f"fpfs{tag}_m00_err"
            if base_column_name is not None:
                err_name = base_column_name + err_name
            err_block = np.empty(
                n_rows, dtype=np.dtype([(err_name, np.float64)]),
            )
            err_block[err_name] = linear_modes[std_key]
            blocks.append(err_block)
    return rfn.merge_arrays(blocks, flatten=True, usemask=False)


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
    detection: NDArray | None = None,
    psf_object: BasePsf | None | NDArray = None,
    do_compute_detect_weight: bool = True,
    return_only_linear_modes: bool = False,
    pack_linear_modes: bool = False,
    base_column_name: str | None = None,
    **kwargs,
):
    """Run the full FPFS measurement pipeline on an exposure.

    Performs source detection (unless a pre-computed catalogue is
    provided), measures shapelet and detection modes, applies non-linear
    shear estimators, and returns a structured catalogue.

    Args:
        fpfs_config (FpfsConfig): Configuration object holding all
            tuneable FPFS parameters.
        pixel_scale (float): Pixel scale in arcseconds.
        noise_variance (float): Variance of image noise per pixel.
        mag_zero (float): Magnitude zero-point of the exposure.
        gal_array (NDArray): Galaxy exposure array.
        psf_array (NDArray): Average PSF image of shape ``(npix, npix)``.
        noise_array (NDArray | None): Pure noise array for noise-bias
            subtraction.
        mask_array (NDArray | None): Mask array (1 for masked pixels).
        detection (NDArray | None): Pre-computed detection catalogue with
            ``('y', 'x')`` columns.  If ``None``, detection is run
            internally.
        psf_object (BasePsf | None | NDArray): Spatially varying PSF
            model.  Falls back to *psf_array* when ``None``.
        do_compute_detect_weight (bool): Whether to measure shapelet
            modes at the default kernel scale.
        return_only_linear_modes (bool): If ``True``, return raw linear
            modes instead of the non-linear shear catalogue.
        pack_linear_modes (bool): When *return_only_linear_modes* is
            ``True``, pack results into a single structured array.
        base_column_name (str | None): Optional prefix prepended to
            every output column name.

    Returns:
        NDArray | dict: Structured FPFS catalogue (default), or a dict
        of linear-mode arrays when *return_only_linear_modes* is ``True``.
    """
    ratio = 1.0 / (10 ** ((30 - mag_zero) / 2.5))
    r2_min = fpfs_config.r2_min * ratio
    omega_r2 = fpfs_config.omega_r2 * ratio
    v_min = fpfs_config.v_min * ratio
    omega_v = fpfs_config.omega_v * ratio
    fpfs_c0 = fpfs_config.c0 * ratio

    if psf_object is None:
        psf_object = psf_array

    linear_modes: dict[str, np.ndarray] | None = {} \
        if return_only_linear_modes else None
    out_list: list[np.ndarray] | None = None if return_only_linear_modes else []

    if do_compute_detect_weight or (detection is None):
        ftask = FpfsTask(
            npix=fpfs_config.npix,
            pixel_scale=pixel_scale,
            sigma_shapelets=fpfs_config.sigma_shapelets,
            noise_variance=noise_variance,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            do_detection=True,
            bound=fpfs_config.bound,
        )
        std_m00 = ftask.std_m00
        m00_min = fpfs_config.snr_min * std_m00
        if detection is None:
            detection = ftask.detect(
                gal_array=gal_array,
                fthres=fpfs_config.fthres,
                pthres=fpfs_config.pthres,
                noise_array=noise_array,
                v_min=v_min,
                omega_v=omega_v,
                mask_array=mask_array,
            )
        else:
            colnames = ("y", "x")
            if detection.dtype.names != colnames:
                raise ValueError("detection has wrong column names")

        if linear_modes is not None:
            linear_modes["detection"] = detection
        else:
            assert out_list is not None
            out_list.append(detection)

        if do_compute_detect_weight:
            src = ftask.run(
                gal_array=gal_array,
                psf=psf_object,
                det=detection,
                noise_array=noise_array,
            )
            del ftask
            if linear_modes is not None:
                linear_modes["data"] = src["data"]
                linear_modes["noise"] = src["noise"]
                linear_modes["std_m00"] = std_m00
            else:
                assert out_list is not None
                meas = measure_fpfs(
                    C0=fpfs_c0,
                    v_min=v_min,
                    omega_v=omega_v,
                    pthres=fpfs_config.pthres,
                    m00_min=m00_min,
                    std_m00=std_m00,
                    r2_min=r2_min,
                    omega_r2=omega_r2,
                    x_array=src["data"],
                    y_array=src["noise"],
                )
                meas = rfn.append_fields(
                    meas, "m00_err",
                    np.full(len(meas), std_m00, dtype=np.float64),
                    usemask=False,
                )
                out_list.append(_append_flux_gauss(
                    meas, fpfs_config.sigma_shapelets, std_m00, "fpfs",
                ))
            del src

    if detection is None and (
        fpfs_config.sigma_shapelets1 > 0 or fpfs_config.sigma_shapelets2 > 0
    ):
        raise RuntimeError("Do not have detection array")

    if fpfs_config.sigma_shapelets1 > 0:
        ftask = FpfsTask(
            npix=fpfs_config.npix,
            pixel_scale=pixel_scale,
            sigma_shapelets=fpfs_config.sigma_shapelets1,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            do_detection=False,
        )
        ftask.prepare_covariance(variance=noise_variance)
        std_m00_1 = ftask.std_m00
        src = ftask.run(
            gal_array=gal_array,
            psf=psf_object,
            det=detection,
            noise_array=noise_array,
        )
        del ftask
        if linear_modes is not None:
            linear_modes["data1"] = src["data"]
            linear_modes["noise1"] = src["noise"]
            linear_modes["std_m00_1"] = std_m00_1
        else:
            meas1 = measure_fpfs(
                C0=fpfs_c0,
                x_array=src["data"],
                y_array=src["noise"],
            )
            meas1 = rfn.append_fields(
                meas1, "m00_err",
                np.full(len(meas1), std_m00_1, dtype=np.float64),
                usemask=False,
            )
            out_list.append(_append_flux_gauss(
                meas1, fpfs_config.sigma_shapelets1, std_m00_1, "fpfs1",
            ))
            del meas1
        del src

    if fpfs_config.sigma_shapelets2 > 0:
        ftask = FpfsTask(
            npix=fpfs_config.npix,
            pixel_scale=pixel_scale,
            sigma_shapelets=fpfs_config.sigma_shapelets2,
            psf_array=psf_array,
            kmax_thres=fpfs_config.kmax_thres,
            do_detection=False,
        )
        ftask.prepare_covariance(variance=noise_variance)
        std_m00_2 = ftask.std_m00
        src = ftask.run(
            gal_array=gal_array,
            psf=psf_object,
            det=detection,
            noise_array=noise_array,
        )
        del ftask
        if linear_modes is not None:
            linear_modes["data2"] = src["data"]
            linear_modes["noise2"] = src["noise"]
            linear_modes["std_m00_2"] = std_m00_2
        else:
            meas2 = measure_fpfs(
                C0=fpfs_c0,
                x_array=src["data"],
                y_array=src["noise"],
            )
            meas2 = rfn.append_fields(
                meas2, "m00_err",
                np.full(len(meas2), std_m00_2, dtype=np.float64),
                usemask=False,
            )
            out_list.append(_append_flux_gauss(
                meas2, fpfs_config.sigma_shapelets2, std_m00_2, "fpfs2",
            ))
            del meas2
        del src

    if out_list is not None:
        result = rfn.merge_arrays(
            out_list,
            flatten=True,
            usemask=False,
        )
        if base_column_name is not None:
            assert result.dtype.names is not None
            map_dict = {
                name: base_column_name + name for name in result.dtype.names
            }
            result = rfn.rename_fields(result, map_dict)
        return result
    else:
        if not pack_linear_modes:
            return linear_modes
        else:
            assert linear_modes is not None
            return _pack_linear_modes(
                linear_modes,
                base_column_name=base_column_name,
            )


__all__ = [
    "measure_fpfs_wsel", "measure_fpfs_wdet", "measure_fpfs_shape",
    "measure_fpfs_wdet0", "measure_shapelets_dg", "fpfs_cut_sigma_ratio",
    "fpfs_det_sigma2",
]
