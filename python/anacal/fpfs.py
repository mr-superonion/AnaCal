from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.lib.recfunctions as rfn
from numpy.typing import NDArray

from ._anacal.fpfs import (
    FpfsImage,
)
from ._anacal.fpfs import ForceTask as _ForceTask
from ._anacal.fpfs import catalog_columns as _catalog_columns
from ._anacal.fpfs import gauss_kernel_rfft as _gauss_kernel_rfft
from ._anacal.fpfs import get_kmax as _get_kmax
from ._anacal.fpfs import m00_to_flux as _m00_to_flux
from ._anacal.fpfs import (
    measure_fpfs,
    measure_fpfs_shape,
    measure_shapelets_dg,
)
from ._anacal.fpfs import shapelets2d as _shapelets2d
from ._anacal.fpfs import shapelets2d_func as _shapelets2d_func
from ._anacal.image import Image
from ._anacal.task import THRESHOLD_REF_MAG_ZERO
from .psf import BasePsf

# Lower bound on the per-kernel flux uncertainty.  A noiseless image gives
# std_m00 == 0, which would make s2n and its shear response infinite; this
# floor keeps those columns finite without affecting any real measurement.
FLUX_ERR_MIN = 1e-8


norder_shapelets = 6


@lru_cache(maxsize=16)
def _fpfs_bases(norder: int, npix: int, sigma: float, kmax: float):
    """Shapelet basis for :class:`FpfsTask`, cached across constructions.

    The basis is a pure function of these four arguments, but callers
    (e.g. xlens) construct an ``FpfsTask`` per band per cell, so without
    the cache the 49 Laguerre modes are recomputed thousands of times per
    patch.  The returned arrays are shared between instances and only ever
    read, which the writeable=False flag enforces.
    """
    sfunc, snames = shapelets2d(
        norder=norder, npix=npix, sigma=sigma, kmax=kmax
    )
    bfunc = np.vstack([sfunc])
    # C-contiguous copy: measure_source reads this filter once per source
    # per k-pixel, and the contiguous layout lets the C++ side walk it
    # with a flat pointer instead of strided access.
    bfunc_use = np.ascontiguousarray(np.transpose(bfunc, (1, 2, 0)))
    bfunc.setflags(write=False)
    bfunc_use.setflags(write=False)
    return bfunc, bfunc_use, tuple(snames)


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


# Flux-family columns carry the kernel token
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


@lru_cache(maxsize=64)
def _catalog_dtype(kernel: str, base_column_name: str | None):
    """FINAL output dtype for one kernel's forced catalog.

    Built once per (kernel, band prefix) from the canonical C++ field
    order (``catalog_columns``), with the kernel token and band prefix
    already applied -- ForceTask writes straight into it, so the catalog
    is born with the right column names.
    """
    mapping = _kernel_rename_map(_catalog_columns, kernel)
    prefix = base_column_name or ""
    return np.dtype(
        [(prefix + mapping[name], "<f8") for name in _catalog_columns]
    )


def _measure_kernel_catalog(
    *,
    fpfs_config,
    pixel_scale,
    gal_array,
    noise_array,
    detection,
    psf_object,
    psf_array,
    noise_variance,
    fpfs_c0,
    sigma_shapelets,
    kernel,
    base_column_name,
    mask_value=None,
    mask_value_max=None,
    psf_model=None,
    psf_offset=(0.0, 0.0),
):
    """One kernel's finished forced catalog via the C++ ``ForceTask``.

    Mirrors ``task.Task.process_image``: setup under the GIL (the cached
    shapelet basis, the covariance scalar ``std_m00`` and -- for a
    spatially varying PSF model -- the per-source PSF stamps), then a
    single C++ pass over the sources with the GIL released, writing
    finished rows straight into the final dtype.  Row values are
    bit-identical to the original numpy assembly.
    """
    ftask = FpfsTask(
        npix=fpfs_config.npix,
        pixel_scale=pixel_scale,
        sigma_shapelets=sigma_shapelets,
        psf_array=psf_array,
        kmax_thres=fpfs_config.kmax_thres,
    )
    ftask.prepare_covariance(variance=noise_variance)
    klim = 1e10 if ftask.kmax is None else (ftask.kmax / pixel_scale)
    force = _ForceTask(
        fpfs_config.npix,
        pixel_scale,
        sigma_shapelets,
        klim,
        fpfs_c0,
    )
    if psf_model is not None:
        # Per-source PSFs are drawn NATIVELY inside the C++ ForceTask
        # loop (psf.PerSourcePsf); psf_array is the per-cell /
        # patch-level 2-D stamp and is not used for the deconvolution.
        psf_use = np.ascontiguousarray(psf_array, dtype=np.float64)
    elif isinstance(psf_object, np.ndarray) and psf_object.ndim == 2:
        psf_use = np.ascontiguousarray(psf_object, dtype=np.float64)
    else:
        raise ValueError(
            "Python-side per-source PSF drawing has been removed: pass "
            "a native psf_model (a psf.PerSourcePsf, or a BasePsf "
            "exposing .native_model), or a single 2-D per-cell PSF "
            "stamp."
        )
    eff_mask = (
        None if mask_value is None
        else np.ascontiguousarray(mask_value, dtype=np.int32)
    )
    return force.process_image(
        gal_array=gal_array,
        psf_array=psf_use,
        filter_image=ftask.bfunc_use,
        det=detection,
        std_m00=float(ftask.std_m00),
        out_dtype=_catalog_dtype(kernel, base_column_name),
        noise_array=noise_array,
        mask_value=eff_mask,
        mask_value_max=mask_value_max,
        psf_model=psf_model,
        psf_offset_x=float(psf_offset[0]),
        psf_offset_y=float(psf_offset[1]),
    )


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

    Prepares the shapelet basis functions, then measures galaxy shapes at
    detected positions via PSF-deconvolved Fourier-space filtering.
    Source DETECTION is not done here: all detection goes through the
    AnaCal detector (``anacal.task.Task`` / ``detector.h``), and the
    resulting catalog is passed in as ``det``.  Call
    :meth:`prepare_covariance` before reading ``std_m00`` / ``std_modes``.

    Args:
        npix (int): Number of pixels per side of a postage stamp.
        pixel_scale (float): Pixel scale in arcseconds.
        sigma_shapelets (float): Shapelet Gaussian kernel size in arcseconds.
            Must be less than 3.0.
        kmax (float | None): Maximum wavenumber for Fourier-space truncation.
            If ``None``, estimated automatically from the PSF.
        psf_array (NDArray | None): Average PSF image of shape
            ``(npix, npix)``.  Defaults to a delta function at centre.
        kmax_thres (float): Threshold for automatic ``kmax`` estimation.
    """

    def __init__(
        self,
        *,
        npix: int,
        pixel_scale: float,
        sigma_shapelets: float,
        kmax: float | None = None,
        psf_array: NDArray | None = None,
        kmax_thres: float = 1e-20,
    ) -> None:
        self.npix = npix

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

        return

    def prepare_fpfs_bases(self):
        """Prepare the FPFS shapelet bases (cached, see _fpfs_bases)."""

        self.bfunc, self.bfunc_use, snames = _fpfs_bases(
            norder_shapelets,
            self.npix,
            float(self.sigmaf),
            float(self.kmax),
        )
        self.colnames = list(snames)
        self.ncol = len(self.colnames)
        self.dtype = [(name, "f8") for name in self.colnames]
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

    FPFS only MEASURES here: detection (and the detection/selection
    weight) lives in the AnaCal detector (``anacal.task.Task`` /
    ``detector.h``), so this config carries no detection thresholds.

    ``c0`` is specified in units that assume a magnitude zero-point of
    ``THRESHOLD_REF_MAG_ZERO`` (the fixed AB nanojansky zeropoint that the
    measurement normalizes every image onto).  It is rescaled internally
    by :func:`process_image` to match the actual zero-point, which is a
    no-op on that path.
    """

    npix: int = 64
    """size of the stamp before Fourier Transform"""

    kmax_thres: float = 1e-12
    """The threshold used to define the upper limit of k we use in Fourier
    space."""

    sigma_shapelets1: float = -1
    """Smoothing scale of the first shapelet kernel.  REQUIRED (> 0) for
    shear estimation."""

    sigma_shapelets2: float = -1
    """Smoothing scale of the second shapelet kernel (optional; <= 0
    disables it)."""

    c0: float = 30.0
    """Weighting parameter for m00 for ellipticity definition (flux scale,
    at THRESHOLD_REF_MAG_ZERO)."""


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
    base_column_name: str | None = None,
    mask_value: NDArray | None = None,
    mask_value_max: int | None = None,
    psf_model=None,
    psf_offset: tuple = (0.0, 0.0),
    **kwargs,
):
    """Run the full FPFS measurement pipeline on an exposure.

    Measures shapelet modes at the given detected positions with the
    ``sigma_shapelets1`` kernel (required) and, when set, the
    ``sigma_shapelets2`` kernel, applies the non-linear shear estimators,
    and returns a structured catalogue.  Detection is NOT done here: all
    detection goes through the AnaCal detector (``anacal.task.Task`` /
    ``detector.h``), and its catalogue must be supplied via
    ``detection``.

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
        detection (NDArray): Pre-computed detection catalogue with
            ``('y', 'x')`` columns (pixel positions), e.g. converted from
            the ``x1_det``/``x2_det`` columns of an ``anacal.task.Task``
            detection.  Required.
        psf_object (BasePsf | None | NDArray): Spatially varying PSF
            model.  Falls back to *psf_array* when ``None``.
        base_column_name (str | None): Optional prefix prepended to
            every output column name.

    Returns:
        NDArray: Structured FPFS catalogue.
    """
    # The flux-scale ellipticity weight c0 is defined at
    # THRESHOLD_REF_MAG_ZERO -- the fixed AB nanojansky zeropoint that the
    # measurement normalizes every image onto -- so this ratio is exactly
    # 1.0 on that path.  It only bites for callers feeding an image on its
    # native zeropoint (e.g. Euclid VIS MAGZERO=24.6).
    ratio = 10 ** ((mag_zero - THRESHOLD_REF_MAG_ZERO) / 2.5)
    fpfs_c0 = fpfs_config.c0 * ratio

    if psf_object is None:
        psf_object = psf_array
    if psf_model is None and psf_object is not None:
        # BasePsf adapters carrying a native handle (NativeCoaddPsf,
        # GridPsf): per-source drawing happens in C++, never in Python.
        # ``native_model`` is a property that BUILDS the model, so it is
        # fetched explicitly rather than probed with hasattr(), which
        # would swallow a real construction error and then fail later
        # with a misleading "no per-source PSF" message.
        psf_model = getattr(type(psf_object), "native_model", None)
        if psf_model is not None:
            psf_model = psf_object.native_model
            psf_offset = (
                float(getattr(psf_object, "x_min", 0.0)),
                float(getattr(psf_object, "y_min", 0.0)),
            )

    if detection is None:
        raise ValueError(
            "process_image requires a detection catalogue: run detection "
            "with the AnaCal detector (anacal.task.Task / detector.h) and "
            "pass its positions as a structured array with ('y', 'x') "
            "columns."
        )
    if detection.dtype.names != ("y", "x"):
        raise ValueError("detection has wrong column names")
    if mask_value is not None:
        if len(mask_value) != len(detection):
            raise ValueError(
                "mask_value must hold one value per detection"
            )
        if psf_model is not None and (
            mask_value.dtype != np.int32
            or not mask_value.flags["C_CONTIGUOUS"]
        ):
            # With a per-source PSF model the C++ writes the 414
            # sentinel INTO this array; anything that needs converting
            # would be written to a temporary and silently lost.
            raise ValueError(
                "mask_value must be a C-contiguous int32 array when "
                "psf_model is given: it is updated in place with the "
                "PSF-invalid sentinel"
            )
    if not fpfs_config.sigma_shapelets1 > 0:
        raise ValueError(
            "sigma_shapelets1 must be set (> 0): it is the measurement "
            "kernel required for shear estimation."
        )

    def measure(sigma_shapelets, kernel):
        return _measure_kernel_catalog(
            fpfs_config=fpfs_config,
            pixel_scale=pixel_scale,
            gal_array=gal_array,
            noise_array=noise_array,
            detection=detection,
            psf_object=psf_object,
            psf_array=psf_array,
            noise_variance=noise_variance,
            fpfs_c0=fpfs_c0,
            sigma_shapelets=sigma_shapelets,
            kernel=kernel,
            base_column_name=base_column_name,
            mask_value=mask_value,
            mask_value_max=mask_value_max,
            psf_model=psf_model,
            psf_offset=psf_offset,
        )

    # One GIL-released C++ pass per kernel (fpfs.ForceTask, structured
    # like task.Task) writes finished rows straight into the final
    # column names -- the band prefix included -- so no numpy column
    # surgery or renaming happens here.
    out_list = [measure(fpfs_config.sigma_shapelets1, "fpfs1")]
    if fpfs_config.sigma_shapelets2 > 0:
        out_list.append(measure(fpfs_config.sigma_shapelets2, "fpfs2"))

    # A single kernel needs no merge; two kernels keep the original rfn
    # merge (values and column order are unchanged either way).
    if len(out_list) == 1:
        return out_list[0]
    return rfn.merge_arrays(out_list, flatten=True, usemask=False)


__all__ = [
    "measure_fpfs", "measure_fpfs_shape", "measure_shapelets_dg",
]
