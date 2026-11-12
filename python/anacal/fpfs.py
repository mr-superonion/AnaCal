from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.lib.recfunctions as rfn
from numpy.typing import NDArray

from ._anacal.fpfs import ForceTask as _ForceTask
from ._anacal.fpfs import (
    FpfsImage,
)
from ._anacal.fpfs import adaptive_catalog_columns as _adaptive_columns
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
from ._anacal import renoise as renoise_mod
from ._anacal.image import Image
from ._anacal.task import THRESHOLD_REF_MAG_ZERO
from ._anacal.task import KLIM_PIX as _KLIM_PIX
from .psf import BasePsf

# Lower bound on the per-kernel flux uncertainty.  A noiseless image gives
# std_m00 == 0, which would make s2n and its shear response infinite; this
# floor keeps those columns finite without affecting any real measurement.
FLUX_ERR_MIN = 1e-8


# Shear-estimator order -> highest shapelet order of the basis.  The
# 2nd-order estimator e = m22 / (m00 + c0) needs m40 and m44 for de/dg
# and m42 for the trace-size response dm20/dg = -sqrt(6) m42, so its
# basis runs through m44 (9 modes); the 4th-order estimator
# q = m42 / (m00 + c0) needs m60 and m64 for dq/dg, so its basis runs
# through m64 (12 modes).
SHAPE_ORDER_DEFAULT = 2
_SHAPELET_NORDER = {2: 4, 4: 6}


def shapelet_norder(shape_order: int) -> int:
    """Highest shapelet order of the basis for a shear-estimator order.

    Args:
        shape_order (int): 2 (e, with the trace-size response) or 4
            (e and q).

    Returns:
        int: ``norder`` for :func:`shapelets2d`: 4 (9 modes through
        m44) or 6 (12 modes through m64).
    """
    try:
        return _SHAPELET_NORDER[int(shape_order)]
    except (KeyError, ValueError, TypeError):
        raise ValueError(
            f"shape_order must be 2 or 4, got {shape_order!r}"
        ) from None


# ---------------------------------------------------------------------
# Adaptive kernel ("afpfs1"): every source is measured with its own
# kernel size, derived from the detector's Gaussian fit.
# ---------------------------------------------------------------------
ADAPTIVE_KERNEL = "afpfs1"
# the second adaptive kernel, measured when adaptive_fpfs_kernel_ratio
# >= ADAPTIVE_RATIO_MIN: sigma^2 = floor^2 + ratio^2 T / 2
ADAPTIVE_KERNEL2 = "afpfs2"
ADAPTIVE_RATIO_MIN = 1.1
# detection-catalog columns the adaptive kernel size is built from (the
# intrinsic covariance of the Gaussian fit, arcsec^2, with its shear
# response; see table.h galRow)
ADAPTIVE_FIT_COLUMNS = (
    "mxx", "myy", "dmxx_dg1", "dmxx_dg2", "dmyy_dg1", "dmyy_dg2"
)
# relative width of the smooth clamps of the size map (in sigma^2)
_ADAPTIVE_CLAMP_EPS = 0.02


def _smooth_max(y, lo, eps):
    """``lo`` for y << lo, ``y`` for y >> lo, C-infinity in between;
    returns the value and d/dy (ngmix rmodel.h smooth_max)."""
    d = y - lo
    r = np.sqrt(d * d + eps * eps)
    return lo + 0.5 * (d + r), 0.5 * (1.0 + d / r)


def _smooth_min(y, hi, eps):
    """``y`` for y << hi, ``hi`` for y >> hi; value and d/dy."""
    d = y - hi
    r = np.sqrt(d * d + eps * eps)
    return hi + 0.5 * (d - r), 0.5 * (1.0 - d / r)


def adaptive_sigma(
    mxx, myy, dmxx_dg1, dmxx_dg2, dmyy_dg1, dmyy_dg2,
    sigma_floor: float, sigma_max: float, ratio: float = 1.0,
):
    """Per-source kernel size of the adaptive measurement.

    sigma^2 = sigma_floor^2 + ratio^2 T / 2 with T = mxx + myy the trace
    of the fit's intrinsic (pre-seeing) covariance, i.e. the galaxy's
    own variance along one axis (times ``ratio``^2; 1 for afpfs1, the
    configured ``adaptive_fpfs_kernel_ratio`` for afpfs2) is added to
    the kernel's in quadrature.  A smooth floor at ``sigma_floor`` and a
    smooth cap at ``sigma_max`` keep sigma inside (sigma_floor,
    sigma_max) with a derivative everywhere, so the shear response
    dsigma/dg (from the fit's dmxx_dg, dmyy_dg) is that of the size
    actually used.

    Returns:
        tuple[NDArray, NDArray, NDArray]: ``(sigma, dsigma_dg1,
        dsigma_dg2)`` in arcsec.
    """
    s0sq = float(sigma_floor) ** 2
    smsq = float(sigma_max) ** 2
    half = 0.5 * float(ratio) ** 2
    f64 = np.float64
    tt = np.asarray(mxx, dtype=f64) + np.asarray(myy, dtype=f64)
    y = s0sq + half * tt
    dy1 = half * (np.asarray(dmxx_dg1, f64) + np.asarray(dmyy_dg1, f64))
    dy2 = half * (np.asarray(dmxx_dg2, f64) + np.asarray(dmyy_dg2, f64))
    # cap first, floor last: the floor is then strict (the cap's clamp
    # lowers every value by eps^2 / (4 |y - cap|), which must not push a
    # small source below the floor node of the basis table)
    eps = _ADAPTIVE_CLAMP_EPS * s0sq
    y, d = _smooth_min(y, smsq, eps)
    dy1, dy2 = dy1 * d, dy2 * d
    y, d = _smooth_max(y, s0sq, eps)
    dy1, dy2 = dy1 * d, dy2 * d
    sigma = np.sqrt(y)
    return sigma, dy1 / (2.0 * sigma), dy2 / (2.0 * sigma)


def adaptive_sigma_grid(sigma_floor: float, sigma_max: float, step: float):
    """Node sizes of the adaptive basis table: geometric (uniform in
    log sigma) with relative spacing ``step``, one node below
    ``sigma_floor`` (= node 1) and nodes through and one beyond
    ``sigma_max``, so every size in [sigma_floor, sigma_max] has the
    four nodes its cubic interpolant needs."""
    if not sigma_max > sigma_floor > 0:
        raise ValueError(
            "adaptive kernel: need 0 < sigma_floor < sigma_max, got "
            f"{sigma_floor!r}, {sigma_max!r}"
        )
    if not 0 < step < 1:
        raise ValueError(f"adaptive_grid_step must be in (0, 1), got {step!r}")
    n_int = int(np.ceil(np.log(sigma_max / sigma_floor) / np.log1p(step)))
    return float(sigma_floor) * (1.0 + step) ** np.arange(-1, n_int + 2)


@lru_cache(maxsize=8)
def _adaptive_table(
    norder: int, npix: int, pixel_scale: float, kmax: float, grid: tuple
):
    """Tabulated basis of the adaptive kernel over the size grid.

    Returns ``(table, b00)``: the (n_nodes, nky, nkx, ncol) C-contiguous
    stack of the node bases (what ForceTask.process_image_adaptive
    interpolates on) and the tuple of the nodes' m00 basis functions
    (for the mode-noise table).
    """
    bases = [
        _fpfs_bases(norder, npix, float(pixel_scale / sig), kmax)
        for sig in grid
    ]
    table = np.ascontiguousarray(
        np.stack([b[1] for b in bases], axis=0), dtype=np.complex128
    )
    table.setflags(write=False)
    b00 = tuple(b[0][0] for b in bases)
    return table, b00


@lru_cache(maxsize=256)
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
        norder (int): Maximum shapelet order.  Must be 2, 4, or 6: the
            first 4, 9 or all 12 modes of the list above.  Shear
            estimation needs 4 (2nd-order estimator) or 6 (4th-order
            estimator too); see :func:`shapelet_norder`.
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
    raise ValueError(f"norder must be 2, 4 or 6, got {norder!r}")


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
def _catalog_dtype(
    kernel: str, base_column_name: str | None, adaptive: bool = False
):
    """FINAL output dtype for one kernel's forced catalog.

    Built once per (kernel, band prefix) from the canonical C++ field
    order (``catalog_columns``, or ``adaptive_catalog_columns`` for the
    adaptive kernel: the same fields plus sigma, dsigma_dg1,
    dsigma_dg2), with the kernel token and band prefix already applied
    -- ForceTask writes straight into it, so the catalog is born with
    the right column names.
    """
    columns = _adaptive_columns if adaptive else _catalog_columns
    mapping = _kernel_rename_map(columns, kernel)
    prefix = base_column_name or ""
    return np.dtype([(prefix + mapping[name], "<f8") for name in columns])


_POSITION_DTYPE = np.dtype([("y", np.float64), ("x", np.float64)])


def _positions(detection):
    """The ('y', 'x') record array ForceTask takes: the detection
    itself when it has exactly those fields, else a copy of them."""
    if detection.dtype == _POSITION_DTYPE:
        return detection
    det = np.zeros(len(detection), dtype=_POSITION_DTYPE)
    det["y"] = detection["y"]
    det["x"] = detection["x"]
    return det


def _measure_kernel_catalog(
    *,
    fpfs_config,
    pixel_scale,
    gal_array,
    noise_array,
    noise2_array,
    detection,
    psf_object,
    psf_array,
    noise_variance,
    fpfs_c0,
    sigma_shapelets,
    kernel,
    base_column_name,
    n_mask_base=None,
    n_mask_base_max=None,
    psf_model=None,
    psf_offset=(0.0, 0.0),
    renoise_mode="rotate",
    renoise_c=0.1,
    renoise_delta=0.01,
    noise_corr=None,
    noise_rotated=True,
    adaptive=False,
    adaptive_ratio=1.0,
):
    """One kernel's finished forced catalog via the C++ ``ForceTask``.

    Mirrors ``task.Task.process_image``: setup under the GIL (the cached
    shapelet basis, the covariance scalar ``std_m00`` and -- for a
    spatially varying PSF model -- the per-source PSF stamps), then a
    single C++ pass over the sources with the GIL released, writing
    finished rows straight into the final dtype.  Row values are
    bit-identical to the original numpy assembly.

    With ``adaptive`` the kernel size is per source (``sigma_shapelets``
    is then the floor of the size map, ``adaptive_ratio`` its ratio, see
    :func:`adaptive_sigma`): the basis is tabulated on a size grid, the
    mode-noise scalar is tabulated at the same nodes, and ``detection``
    must carry the fit columns ``ADAPTIVE_FIT_COLUMNS``.
    """
    ftask = FpfsTask(
        npix=fpfs_config.npix,
        pixel_scale=pixel_scale,
        sigma_shapelets=sigma_shapelets,
        psf_array=psf_array,
        kmax=fpfs_config.kmax,
        kmax_thres=fpfs_config.kmax_thres,
        shape_order=fpfs_config.shape_order,
    )
    klim = 1e10 if ftask.kmax is None else (ftask.kmax / pixel_scale)
    if renoise_mode not in ("rotate", "spectral"):
        raise ValueError(
            f"renoise_mode must be 'rotate' or 'spectral', got {renoise_mode!r}"
        )
    renoise = None
    if klim > 1e9:
        # no finite Fourier cutoff: the white-noise covariance of the
        # plain path (image noise plus the rotated copy)
        ftask.prepare_covariance(variance=noise_variance)
    else:
        # The RAW noise spectrum of the patch on the stamp grid, from the
        # correlation function (white without one), in BOTH modes.  For
        # Option 2 ("spectral") the kernels symmetrize this raw spectrum
        # in C++ ONCE per call, with the per-cell ``psf_array`` (the same
        # PSF the detection stage used), for the constant-PSF and the
        # native per-source-PSF paths alike: the renoising fields are one
        # fixed realization shared with detection, so only the
        # deconvolution filter chi / p follows the source PSF.  ForceTask
        # applies all three kernels in Fourier space per stamp: the
        # spin-2 filler rho on the second realization noise2_array, the
        # added noise eta_0 and the assigned responses tau_j on
        # noise_array.  The covariance is summed over the total noise
        # spectrum (image noise plus the rotated copy or the symmetrized
        # spectrum plus the added noise).
        spectrum = renoise_mod.make_noise_spectrum(
            fpfs_config.npix,
            pixel_scale,
            float(noise_variance),
            None if noise_corr is None else np.ascontiguousarray(
                noise_corr, dtype=np.float64
            ),
        )
        mode = "rotate"
        if renoise_mode == "spectral" and noise_array is not None:
            if noise2_array is None:
                raise ValueError(
                    "spectral renoising needs a second noise realization "
                    "(noise2_array): the spin-2 filler is built from it"
                )
            mode = "spectral"
            renoise = renoise_mod.Setup(
                c=renoise_c,
                spectra=[spectrum],
                noise_rotated=noise_rotated,
                mode=mode,
                delta=renoise_delta,
            )
        spectral = mode == "spectral"
        kernels = renoise_mod.make_renoise_kernels(
            spectrum, np.ascontiguousarray(psf_array, dtype=np.float64),
            klim, renoise_c, renoise_delta if spectral else 0.0,
            spectral, noise_rotated, mode,
        )
        ftask.prepare_covariance(
            variance=noise_variance,
            renoise_kernels=kernels,
            with_noise_image=noise_array is not None,
        )
    force = _ForceTask(
        fpfs_config.npix,
        pixel_scale,
        sigma_shapelets,
        klim,
        fpfs_c0,
    )
    if adaptive:
        missing = [
            c for c in ADAPTIVE_FIT_COLUMNS
            if c not in (detection.dtype.names or ())
        ]
        if missing:
            raise ValueError(
                "adaptive kernel: the detection catalog lacks the fit "
                f"columns {missing}"
            )
        sigma_max = fpfs_config.adaptive_sigma_max
        if sigma_max <= 0:
            sigma_max = 3.0 * sigma_shapelets
        if sigma_max > 3.0:
            raise ValueError("adaptive_sigma_max should be < 3 arcsec")
        grid = adaptive_sigma_grid(
            sigma_shapelets, sigma_max, fpfs_config.adaptive_grid_step
        )
        table, b00 = _adaptive_table(
            ftask.norder, fpfs_config.npix, float(pixel_scale),
            float(ftask.kmax), tuple(float(g) for g in grid),
        )
        std_grid = ftask.m00_std_grid(
            b00,
            variance=noise_variance,
            renoise_kernels=None if klim > 1e9 else kernels,
            with_noise_image=noise_array is not None,
        )
        sig, dsig1, dsig2 = adaptive_sigma(
            detection["mxx"], detection["myy"],
            detection["dmxx_dg1"], detection["dmxx_dg2"],
            detection["dmyy_dg1"], detection["dmyy_dg2"],
            sigma_shapelets, sigma_max, adaptive_ratio,
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
        None if n_mask_base is None
        else np.ascontiguousarray(n_mask_base, dtype=np.float32)
    )
    common = dict(
        gal_array=gal_array,
        psf_array=psf_use,
        det=_positions(detection),
        out_dtype=_catalog_dtype(kernel, base_column_name, adaptive),
        noise_array=noise_array,
        noise2_array=(
            None if renoise is None else np.asarray(
                noise2_array, dtype=np.float32
            )
        ),
        n_mask_base=eff_mask,
        n_mask_base_max=n_mask_base_max,
        psf_model=psf_model,
        psf_offset_x=float(psf_offset[0]),
        psf_offset_y=float(psf_offset[1]),
        renoise=renoise,
    )
    if adaptive:
        return force.process_image_adaptive(
            filter_table=table,
            sigma_grid=np.ascontiguousarray(grid, dtype=np.float64),
            std_m00_grid=np.ascontiguousarray(std_grid, dtype=np.float64),
            sigma=np.ascontiguousarray(sig, dtype=np.float64),
            dsigma_dg1=np.ascontiguousarray(dsig1, dtype=np.float64),
            dsigma_dg2=np.ascontiguousarray(dsig2, dtype=np.float64),
            **common,
        )
    return force.process_image(
        filter_image=ftask.bfunc_use,
        std_m00=float(ftask.std_m00),
        **common,
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
            ``(npix, npix)``, centred on pixel ``(npix // 2, npix // 2)``
            (0-based): that pixel is shifted to the FFT origin, so for even
            ``npix`` the centre is half a pixel off the geometric middle
            (GalSim draws there only after ``.shift(0.5 * scale,
            0.5 * scale)``), and a PSF centred elsewhere shifts the
            deconvolved image by the same amount.  Defaults to a delta
            function at that pixel.
        kmax_thres (float): Threshold for automatic ``kmax`` estimation.
        shape_order (int): Order of the shear estimator the basis must
            serve: 2 measures the 9 modes through m44 (e and its
            response, the trace-size response dm20/dg), 4 the 12 modes
            through m64 (q and its response as well).  See
            :func:`shapelet_norder`.
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
        shape_order: int = SHAPE_ORDER_DEFAULT,
    ) -> None:
        self.npix = npix
        self.shape_order = int(shape_order)
        self.norder = shapelet_norder(shape_order)

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
            self.norder,
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
        ratio: NDArray | None = None,
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
            ratio (NDArray | None): optional multiplier of the deconvolved
                noise spectrum, half-plane ``(npix, npix // 2 + 1)``
                (the spectral renoising's S^d_sym / S^d)

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
        if ratio is not None:
            noise_pf_deconv = noise_pf_deconv * ratio

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

    def _noise_weight(
        self,
        variance: float,
        psf_pow: NDArray,
        noise_pf: NDArray | None = None,
        ratio: NDArray | None = None,
    ) -> NDArray:
        """The deconvolved noise power times the half-plane weights,
        as :meth:`calculate_covariance` forms it: the covariance of two
        modes a, b is ``sum(a * weight * conj(b)).real / scale^4``."""
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
        if ratio is not None:
            noise_pf_deconv = noise_pf_deconv * ratio
        _w = np.ones(psf_pow.shape) * 2.0
        _w[:, 0] = 1.0
        _w[:, -1] = 1.0
        return _w * noise_pf_deconv

    def m00_std_grid(
        self,
        b00_list,
        variance: float,
        noise_pf: NDArray | None = None,
        renoise_kernels=None,
        with_noise_image: bool = True,
    ) -> NDArray:
        """Mode-noise scalar ``std_m00`` for a list of m00 basis
        functions (the nodes of the adaptive kernel's size grid), over
        the same total noise spectrum :meth:`prepare_covariance` uses
        with the same arguments.  The basis of this task is not used.
        """
        if renoise_kernels is not None:
            rk = renoise_kernels
            sd = np.asarray(rk.Sd0)
            total = (
                np.asarray(renoise_mod.total_spectrum(rk))
                if with_noise_image else np.asarray(rk.Sd)
            )
            ratio = np.where(
                sd > 0, total / np.where(sd > 0, sd, 1.0), 1.0
            )
            weight = self._noise_weight(
                variance, self.psf_pow, noise_pf, ratio
            )
        else:
            weight = self._noise_weight(variance, self.psf_pow, noise_pf)
            noise_pf_rot = (
                None if noise_pf is None else self._rotate_noise_pf(noise_pf)
            )
            weight = weight + self._noise_weight(
                variance, self._rotate_rfft_pow(self.psf_pow), noise_pf_rot
            )
        out = np.empty(len(b00_list), dtype=np.float64)
        for i, b00 in enumerate(b00_list):
            var00 = np.sum(b00 * weight * np.conjugate(b00)).real
            out[i] = np.sqrt(var00 / self.pixel_scale**4.0)
        return out

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
        self,
        variance: float,
        noise_pf: NDArray | None = None,
        renoise_kernels=None,
        with_noise_image: bool = True,
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

        With ``renoise_kernels`` (anacal.renoise.RenoiseKernels on the
        stamp grid, built from the patch noise spectrum) the covariance is
        summed over the kernels' total noise spectrum instead: image
        noise plus the rotated copy (mode "rotate") or (1 + c) S^d_sym,
        the symmetrized image noise plus the added noise (mode
        "spectral"); with ``with_noise_image=False`` (no pure-noise
        image) the image noise alone.

        Returns:
            NDArray: covariance matrix of shape ``(ncol, ncol)``
        """
        if renoise_kernels is not None:
            # relative to the white noise of the raw variance (what the
            # plain path integrates): the total spectrum of the kernels
            rk = renoise_kernels
            sd = np.asarray(rk.Sd0)
            total = (
                np.asarray(renoise_mod.total_spectrum(rk))
                if with_noise_image else np.asarray(rk.Sd)
            )
            ratio = np.where(
                sd > 0, total / np.where(sd > 0, sd, 1.0), 1.0
            )
            cov_elems = self.calculate_covariance(
                variance=variance,
                psf_pow=self.psf_pow,
                noise_pf=noise_pf,
                ratio=ratio,
            )
            self.std_modes = np.sqrt(np.diagonal(cov_elems))
            self.std_m00 = self.std_modes[self.di["m00"]]
            return cov_elems
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
            psf_array (NDArray): PSF image of shape ``(npix, npix)``,
                centred on pixel ``(npix // 2, npix // 2)`` (0-based).
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

    kmax: float | None = _KLIM_PIX
    """Upper limit of k in Fourier space, in 1 / pixel (``klim`` in
    1 / arcsec is ``kmax / pixel_scale``).  The default is the cutoff of
    the detection / measurement Task (``anacal.task.KLIM_PIX``), so that
    both stages measure the same Fourier modes; None estimates it from
    the PSF with ``kmax_thres`` instead."""

    sigma_shapelets1: float = -1
    """Smoothing scale of the first shapelet kernel.  REQUIRED (> 0) for
    shear estimation."""

    sigma_shapelets2: float = -1
    """Smoothing scale of the second shapelet kernel (optional; <= 0
    disables it)."""

    c0: float = 30.0
    """Weighting parameter for m00 for ellipticity definition (flux scale,
    at THRESHOLD_REF_MAG_ZERO)."""

    adaptive_sigma: bool = False
    """Also measure the adaptive kernel ``afpfs1``: every source with its
    own kernel size sigma^2 = floor^2 + T / 2, T the trace of the
    detector's fitted intrinsic covariance (``mxx + myy`` of the
    detection catalog, which must then carry ``ADAPTIVE_FIT_COLUMNS``),
    with the shear response of the size folded into every response.
    Its columns are those of the fixed kernels plus ``sigma``,
    ``dsigma_dg1``, ``dsigma_dg2``.  The fixed kernels fpfs1 (and
    fpfs2) are measured regardless."""

    adaptive_fpfs_kernel_ratio: float = 1.0
    """Second adaptive kernel ``afpfs2``, measured when this is >=
    ADAPTIVE_RATIO_MIN (1.1) and ``adaptive_sigma`` is on, with
    sigma^2 = floor^2 + ratio^2 T / 2 (same floor, cap and grid as
    afpfs1).  Below 1.1 no afpfs2 is measured."""

    adaptive_sigma_floor: float = -1
    """Floor of the adaptive kernel size [arcsec]; <= 0 uses
    ``sigma_shapelets1``."""

    adaptive_sigma_max: float = -1
    """Cap of the adaptive kernel size [arcsec] (smooth); <= 0 uses
    three times the floor.  Must stay below 3 arcsec."""

    adaptive_grid_step: float = 0.02
    """Relative spacing of the size grid the adaptive basis is tabulated
    on; a source's modes are the cubic (Catmull-Rom, in log size)
    through the four nodes around its size, with the interpolant's
    slope as their size derivative."""

    shape_order: int = SHAPE_ORDER_DEFAULT
    """Order of the shear estimator: 2 measures the 9 shapelet modes
    through m44, enough for e = m22 / (m00 + c0) with its response and
    for the trace-size response dm20/dg = -sqrt(6) m42; the 4th-order
    columns q1, q2, dq*_dg* and dm42*_dg* are then NaN (m42c/s
    themselves are measured).  4 measures the 12 modes through m64 and
    fills q = m42 / (m00 + c0) and its response too.  The catalog
    columns are the same either way."""


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
    noise2_array: NDArray | None = None,
    mask_array: NDArray | None = None,
    detection: NDArray | None = None,
    psf_object: BasePsf | None | NDArray = None,
    base_column_name: str | None = None,
    n_mask_base: NDArray | None = None,
    n_mask_base_max: float | None = None,
    psf_model=None,
    psf_offset: tuple = (0.0, 0.0),
    renoise_mode: str = "rotate",
    renoise_c: float = 0.1,
    renoise_delta: float = 0.01,
    noise_corr: NDArray | None = None,
    noise_rotated: bool = True,
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
        psf_array (NDArray): Average PSF image of shape ``(npix, npix)``,
            centred on pixel ``(npix // 2, npix // 2)`` (0-based); see
            ``FpfsTask``.
        noise_array (NDArray | None): Pure noise array for noise-bias
            subtraction.
        noise2_array (NDArray | None): second, independent noise
            realization eps'' with the same correlation properties
            (spectral renoising only): the spin-2 filler rho * eps'' is
            applied in Fourier space per stamp.
        mask_array (NDArray | None): Mask array (1 for masked pixels).
        detection (NDArray): Pre-computed detection catalogue with
            ``('y', 'x')`` columns (pixel positions), e.g. converted from
            the ``x1_det``/``x2_det`` columns of an ``anacal.task.Task``
            detection.  Required.
        psf_object (BasePsf | None | NDArray): Spatially varying PSF
            model.  Falls back to *psf_array* when ``None``.
        base_column_name (str | None): Optional prefix prepended to
            every output column name.
        renoise_mode (str): noise-bias correction with ``noise_array``:
            ``"rotate"`` (the 90-degree-rotated noise copy, Option 1) or
            ``"spectral"`` (minimal-variance renoising with the spin-2
            filler and the assigned response, Option 2).  Must be the
            same as the detection stage's.
        renoise_c (float): spectral renoising, added-noise fraction.
        renoise_delta (float): spectral renoising, smoothing of the
            spectrum maximum.
        noise_corr (NDArray | None): the (unrotated) noise correlation
            function of the patch, for the spectral renoising; None is
            white noise.
        noise_rotated (bool): whether ``noise_array`` was drawn from the
            90-degree-rotated correlation function (xlens convention).
            ``noise2_array`` is never rotated: the filler must carry the
            image noise's own correlation.

    Returns:
        NDArray: Structured FPFS catalogue (``catalog_columns`` per
        kernel, same layout for both ``fpfs_config.shape_order``; at
        order 2 the 4th-order estimator columns ``q1``, ``q2``,
        ``dq*_dg*`` and ``dm42*_dg*`` are NaN).  With
        ``fpfs_config.adaptive_sigma`` the ``afpfs1`` kernel's columns
        (``adaptive_catalog_columns``) follow those of the fixed
        kernels, then ``afpfs2``'s when ``adaptive_fpfs_kernel_ratio``
        >= 1.1.
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
    names = detection.dtype.names or ()
    if ("y" not in names) or ("x" not in names):
        raise ValueError(
            "detection needs ('y', 'x') pixel columns (plus the fit "
            "columns ADAPTIVE_FIT_COLUMNS for the adaptive kernel)"
        )
    if n_mask_base is not None:
        if len(n_mask_base) != len(detection):
            raise ValueError(
                "n_mask_base must hold one value per detection"
            )
        if psf_model is not None and (
            n_mask_base.dtype != np.float32
            or not n_mask_base.flags["C_CONTIGUOUS"]
        ):
            # With a per-source PSF model the C++ writes the 1.0
            # sentinel INTO this array; anything that needs converting
            # would be written to a temporary and silently lost.
            raise ValueError(
                "n_mask_base must be a C-contiguous float32 array when "
                "psf_model is given: it is updated in place with the "
                "PSF-invalid sentinel"
            )
    if not fpfs_config.sigma_shapelets1 > 0:
        raise ValueError(
            "sigma_shapelets1 must be set (> 0): it is the measurement "
            "kernel required for shear estimation."
        )
    shapelet_norder(fpfs_config.shape_order)

    def measure(sigma_shapelets, kernel, adaptive=False, ratio=1.0):
        return _measure_kernel_catalog(
            fpfs_config=fpfs_config,
            pixel_scale=pixel_scale,
            gal_array=gal_array,
            noise_array=noise_array,
            noise2_array=noise2_array,
            detection=detection,
            psf_object=psf_object,
            psf_array=psf_array,
            noise_variance=noise_variance,
            fpfs_c0=fpfs_c0,
            sigma_shapelets=sigma_shapelets,
            kernel=kernel,
            base_column_name=base_column_name,
            n_mask_base=n_mask_base,
            n_mask_base_max=n_mask_base_max,
            psf_model=psf_model,
            psf_offset=psf_offset,
            renoise_mode=renoise_mode,
            renoise_c=renoise_c,
            renoise_delta=renoise_delta,
            noise_corr=noise_corr,
            noise_rotated=noise_rotated,
            adaptive=adaptive,
            adaptive_ratio=ratio,
        )

    # One GIL-released C++ pass per kernel (fpfs.ForceTask, structured
    # like task.Task) writes finished rows straight into the final
    # column names -- the band prefix included -- so no numpy column
    # surgery or renaming happens here.
    out_list = [measure(fpfs_config.sigma_shapelets1, "fpfs1")]
    if fpfs_config.sigma_shapelets2 > 0:
        out_list.append(measure(fpfs_config.sigma_shapelets2, "fpfs2"))
    ratio = fpfs_config.adaptive_fpfs_kernel_ratio
    if (ratio >= ADAPTIVE_RATIO_MIN) and not fpfs_config.adaptive_sigma:
        raise ValueError(
            "adaptive_fpfs_kernel_ratio >= 1.1 asks for the afpfs2 kernel, "
            "which needs adaptive_sigma=True"
        )
    if fpfs_config.adaptive_sigma:
        floor = fpfs_config.adaptive_sigma_floor
        if floor <= 0:
            floor = fpfs_config.sigma_shapelets1
        out_list.append(measure(floor, ADAPTIVE_KERNEL, adaptive=True))
        if ratio >= ADAPTIVE_RATIO_MIN:
            out_list.append(
                measure(floor, ADAPTIVE_KERNEL2, adaptive=True, ratio=ratio)
            )

    # A single kernel needs no merge; two kernels keep the original rfn
    # merge (values and column order are unchanged either way).
    if len(out_list) == 1:
        return out_list[0]
    return rfn.merge_arrays(out_list, flatten=True, usemask=False)


__all__ = [
    "measure_fpfs", "measure_fpfs_shape", "measure_shapelets_dg",
]
