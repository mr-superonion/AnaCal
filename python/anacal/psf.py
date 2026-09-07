"""PSF interface and native coadd-PSF models.

Two things live here:

* :class:`BasePsf` -- the abstract adapter every survey-specific PSF
  wrapper implements (``draw(x, y) -> stamp``).  :mod:`anacal.fpfs`
  dispatches on it.
* the native models -- load an LSST DM PSF once (under the GIL) and
  evaluate it in pure C++ (GIL-free).  The C++ side
  (``anacal._anacal.psfmodel``) re-implements the DM chain CoaddPsf ->
  WarpedPsf -> PsfexPsf; this module extracts the model parameters from
  a DM ``CoaddPsf`` and fits the per-visit WCS mapping as polynomials,
  which is the only approximation in the pipeline.  Loading is the ONLY
  step that touches DM/Python objects.

Supported single-visit models: PSFEx (HSC-style) and PIFF
PixelGrid+BasisPolynomial (DES, Rubin); the C++ layer keeps them behind
one interface so further models can be added without touching the
warping/coadding code.
"""

from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from ._anacal.psfmodel import (
    CoaddPsfModel,
    GridPsfModel,
    PerSourcePsf,
    PiffModel,
    PsfexModel,
    lanczos_kernel,
    resize_array,
)

__all__ = [
    "BasePsf",
    "PsfexModel",
    "PiffModel",
    "CoaddPsfModel",
    "GridPsfModel",
    "PerSourcePsf",
    "resize_array",
    "lanczos_kernel",
    "load_coadd_psf_model",
    "NativeCoaddPsf",
]

_LANCZOS_ORDERS = {
    "lanczos3": 3,
    "lanczos4": 4,
    "lanczos5": 5,
}


class BasePsf(ABC):
    """Abstract base PSF class."""

    def __init__(self):
        return

    @abstractmethod
    def draw(self, x, y, *args, **kwargs):
        """Draw the PSF image evaluated at position ``(x, y)``.

        Parameters
        ----------
        x : float
            X-coordinate, in pixels, at which to evaluate the PSF.
        y : float
            Y-coordinate, in pixels, at which to evaluate the PSF.
        *args
            Optional positional arguments forwarded to subclass
            implementations.
        **kwargs
            Optional keyword arguments forwarded to subclass
            implementations.

        Returns
        -------
        numpy.ndarray
            Array representing the PSF image. Subclasses should document
            the expected shape and any normalization of the returned
            array.

        Notes
        -----
        This method is abstract and must be implemented by subclasses.
        """
        pass


class _MappingGrids:
    """Shared fit/check grids over the coadd bbox.

    The coadd->sky transform is identical for every component, so the
    grid sky positions are computed ONCE and only the per-visit
    ``skyToPixelArray`` runs per component; the polynomial design
    matrices per order are cached too.
    """

    def __init__(self, coadd_wcs, bbox, ngrid: int):
        x0, y0 = float(bbox.getMinX()), float(bbox.getMinY())
        x1, y1 = float(bbox.getMaxX()), float(bbox.getMaxY())
        self.cx, self.cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        self.sx, self.sy = 0.5 * (x1 - x0), 0.5 * (y1 - y0)
        g = np.linspace(0.0, 1.0, ngrid)
        h = (g[:-1] + g[1:]) / 2.0  # check grid: never the fit nodes
        gx, gy = np.meshgrid(x0 + g * (x1 - x0), y0 + g * (y1 - y0))
        hx, hy = np.meshgrid(x0 + h * (x1 - x0), y0 + h * (y1 - y0))
        self.gx, self.gy = gx.ravel(), gy.ravel()
        self.hx, self.hy = hx.ravel(), hy.ravel()
        self.g_ra, self.g_dec = coadd_wcs.pixelToSkyArray(
            np.ascontiguousarray(self.gx), np.ascontiguousarray(self.gy)
        )
        self.h_ra, self.h_dec = coadd_wcs.pixelToSkyArray(
            np.ascontiguousarray(self.hx), np.ascontiguousarray(self.hy)
        )
        self._design: dict = {}

    def design(self, order: int, check: bool):
        key = (order, check)
        if key not in self._design:
            xx = self.hx if check else self.gx
            yy = self.hy if check else self.gy
            xn = (xx - self.cx) / self.sx
            yn = (yy - self.cy) / self.sy
            self._design[key] = np.vstack([
                xn**a * yn**b
                for a in range(order + 1)
                for b in range(order + 1)
            ]).T
        return self._design[key]


def _fit_mapping(grids: _MappingGrids, visit_wcs, order: int):
    """Fit one visit's coadd->visit pixel mapping as tensor polynomials.

    Returns (cu, cv, max_residual); the residual is measured in visit
    pixels on the shifted check grid.
    """
    u, v = visit_wcs.skyToPixelArray(grids.g_ra, grids.g_dec)
    A = grids.design(order, check=False)
    cu, *_ = np.linalg.lstsq(A, u, rcond=None)
    cv, *_ = np.linalg.lstsq(A, v, rcond=None)
    uh, vh = visit_wcs.skyToPixelArray(grids.h_ra, grids.h_dec)
    Ah = grids.design(order, check=True)
    res = max(
        float(np.max(np.abs(Ah @ cu - uh))),
        float(np.max(np.abs(Ah @ cv - vh))),
    )
    m = order + 1
    return cu.reshape(m, m), cv.reshape(m, m), res


# The K_m depend on nothing but the Lanczos order, while a coadd hands
# us one component per contributing visit -- all of which use the same
# interpolant (121 identical evaluations on a DP1 patch).
@lru_cache(maxsize=None)
def _lanczos_dc_kvals_cached(n: int, nterm: int = 5) -> tuple:
    """conserve_dc corrections of a Lanczos-``n`` kernel, as in galsim.

    galsim (``Interpolant.cpp``) makes the kernel conserve a flat field
    by dividing the raw ``sinc(x) sinc(x/n)`` by ``1 - 2 sum_{m=1}^{5}
    K_m (1 - cos(2 pi m x))``, where ``K_m`` is the Fourier transform
    of the raw, truncated kernel at integer frequency ``m``::

        K_m = int_{-n}^{n} sinc(x) sinc(x/n) cos(2 pi m x) dx

    The integrand is entire, so Gauss-Legendre quadrature converges to
    machine precision; the kernel built from these K_m matches
    ``galsim.Lanczos(n, conserve_dc=True).xval`` to ~1e-14 for n = 3
    to 11 (``tests/test_psf.py``).  galsim itself is not needed.
    """
    if n < 1:
        raise ValueError(f"Lanczos order must be positive, got {n}")
    # the integrand oscillates at up to ~6 cycles per pixel over 2n
    # pixels; 64 nodes per pixel is far past the exponential-convergence
    # threshold (~36 per pixel) and still trivially cheap
    nodes, weights = np.polynomial.legendre.leggauss(64 * n + 64)
    x = nodes * n
    w = weights * n
    raw = np.sinc(x) * np.sinc(x / n)
    return tuple(
        float(np.sum(w * raw * np.cos(2.0 * np.pi * m * x)))
        for m in range(1, nterm + 1)
    )


def _lanczos_dc_kvals(n: int) -> NDArray:
    """Cached :func:`_lanczos_dc_kvals_cached` as a fresh array.

    The cache stores a tuple so a caller cannot mutate the shared
    entry; each call gets its own array to hand to the C++ model.
    """
    return np.array(_lanczos_dc_kvals_cached(int(n)), dtype=float)


def _load_psfex_component(inner, index: int) -> PsfexModel:
    sd = inner.getSerializationData()
    degree = list(sd.degree)
    group = list(sd.group)
    if len(degree) != 1 or any(g != 0 for g in group):
        raise NotImplementedError(
            "only ndim=2, ngroup=1 PSFEx models are supported "
            f"(component {index}: degree={degree}, group={group})"
        )
    w, h, nbasis = [int(s) for s in sd.size]
    context = np.asarray(sd.context, dtype=float)
    return PsfexModel(
        comp=np.ascontiguousarray(sd.comp, dtype=np.float32),
        w=w,
        h=h,
        nbasis=nbasis,
        pixstep=float(sd.pixel_step),
        degree=int(degree[0]),
        context_offset=np.ascontiguousarray(context[:, 0]),
        context_scale=np.ascontiguousarray(context[:, 1]),
    )


def _load_piff_component(inner, index: int) -> PiffModel:
    # The piff / galsim classes are recognised BY NAME rather than with
    # isinstance: anacal must not import piff, galsim, nor the LSST
    # stack.  Every attribute read below is duck typed for the same
    # reason.

    pp = inner._piffResult
    model = pp.model
    interp = pp.interp
    if type(model).__name__ != "PixelGrid":
        raise NotImplementedError(
            f"component {index}: piff model {type(model).__name__} "
            "is not a PixelGrid"
        )
    if type(interp).__name__ != "BasisPolynomial":
        raise NotImplementedError(
            f"component {index}: piff interp {type(interp).__name__} "
            "is not a BasisPolynomial"
        )
    if "colorValue" in pp.interp_property_names:
        raise NotImplementedError(
            f"component {index}: colour-dependent piff models are not "
            "supported"
        )
    if pp.pointing is not None:
        raise NotImplementedError(
            f"component {index}: piff models with a pointing (sky "
            "coordinates) are not supported"
        )
    wcs = list(pp.wcs.values())[0]
    if type(wcs).__name__ != "PixelScale":
        raise NotImplementedError(
            f"component {index}: piff wcs {type(wcs).__name__} is not "
            "a PixelScale"
        )
    if model._fit_flux:
        raise NotImplementedError(
            f"component {index}: fit_flux piff models are not supported"
        )
    keys = tuple(interp._keys)
    if keys == ("u", "v"):
        coord_scale = float(wcs.scale)
        if abs(model.scale / wcs.scale - 1.0) > 1.0e-12:
            raise NotImplementedError(
                f"component {index}: oversampled piff model "
                f"(model scale {model.scale} != image scale {wcs.scale})"
            )
    elif keys == ("x", "y"):
        coord_scale = 1.0
    else:
        raise NotImplementedError(
            f"component {index}: unsupported interp keys {keys}"
        )
    gs_interp = model.interp
    if type(gs_interp).__name__ != "Lanczos":
        raise NotImplementedError(
            f"component {index}: interpolant "
            f"{type(gs_interp).__name__} is not Lanczos"
        )
    if gs_interp.conserve_dc:
        dc = _lanczos_dc_kvals(int(gs_interp.n))
    else:
        dc = np.zeros(0)
    nmodel = int(model.size)
    q = np.ascontiguousarray(interp.q, dtype=float)
    pairs = np.argwhere(interp._mask)
    if q.shape != (nmodel * nmodel, len(pairs)):
        raise RuntimeError(
            f"component {index}: q shape {q.shape} does not match "
            f"model size {nmodel} and {len(pairs)} terms"
        )
    return PiffModel(
        q=q,
        term_i=np.ascontiguousarray(pairs[:, 0], dtype=np.int32),
        term_j=np.ascontiguousarray(pairs[:, 1], dtype=np.int32),
        nmodel=nmodel,
        stamp=int(inner.width),
        coord_scale=coord_scale,
        lanczos_n=int(gs_interp.n),
        dc_kval=np.ascontiguousarray(dc),
    )


def load_coadd_psf_model(
    coadd_psf,
    coadd_bbox,
    warping_kernel_name: str,
    cache_size: int,
    mapping_tol: float = 1.0e-8,
    mapping_ngrid: int = 25,
) -> CoaddPsfModel:
    """Build the native model from a DM ``CoaddPsf``.

    Parameters
    ----------
    coadd_psf : lsst.meas.algorithms.CoaddPsf
        The DM coadd PSF (``exposure.getPsf()``).
    coadd_bbox : lsst.geom.Box2I
        Region (coadd pixels) over which the WCS mappings are fitted --
        the exposure bbox.  Evaluations outside it lose accuracy.
    warping_kernel_name, cache_size :
        Read from the persisted CoaddPsf by the caller (see
        ``xlens.utils.image.psf._coadd_psf_config``; reading them needs
        DM-archive-format knowledge, which is why it does not live
        here).  They must NOT be assumed: DP1/HSC files carry
        ``lanczos3``/0 and ``lanczos5``/10000, while the DM
        constructor defaults are ``lanczos3``/10000 -- neither file
        matches the default pair.
    mapping_tol : float
        Required max mapping residual in visit pixels; the polynomial
        order is raised until the residual on a check grid is below
        this (or order 10 is reached, which raises).
    """
    if warping_kernel_name not in _LANCZOS_ORDERS:
        raise ValueError(
            f"unsupported warping kernel '{warping_kernel_name}'"
        )
    model = CoaddPsfModel(
        lanczos_order=_LANCZOS_ORDERS[warping_kernel_name],
        cache_size=int(cache_size),
    )
    grids = _MappingGrids(
        coadd_psf.getCoaddWcs(), coadd_bbox, mapping_ngrid
    )
    for i in range(coadd_psf.getComponentCount()):
        inner = coadd_psf.getPsf(i)
        if hasattr(inner, "getSerializationData"):
            visit_model = _load_psfex_component(inner, i)
        elif hasattr(inner, "_piffResult"):
            visit_model = _load_piff_component(inner, i)
        else:
            raise NotImplementedError(
                f"component {i}: unsupported single-visit PSF type "
                f"{type(inner).__name__}"
            )

        bbox = coadd_psf.getBBox(i)
        polygon = coadd_psf.getValidPolygon(i)
        poly_arr = None
        if polygon is not None:
            poly_arr = np.array(
                [(p.getX(), p.getY()) for p in polygon.getVertices()],
                dtype=float,
            )

        visit_wcs = coadd_psf.getWcs(i)
        order = 3
        while True:
            cu, cv, res = _fit_mapping(grids, visit_wcs, order)
            if res < mapping_tol:
                break
            if order >= 10:
                # Some WCS pairs bottom out slightly above the target
                # (lstsq conditioning); anything below 1e-6 visit pixels
                # is still far beyond measurement requirements.
                if res < 1.0e-6:
                    break
                raise RuntimeError(
                    f"component {i}: WCS mapping fit does not reach "
                    f"1e-6 visit pixels (residual {res:.2e} at order 10)"
                )
            order += 1
        model.add_component(
            model=visit_model,
            weight=float(coadd_psf.getWeight(i)),
            bx0=float(bbox.getMinX()) - 0.5,
            by0=float(bbox.getMinY()) - 0.5,
            bx1=float(bbox.getMaxX()) + 0.5,
            by1=float(bbox.getMaxY()) + 0.5,
            polygon=poly_arr,
            map_order=order,
            map_cx=grids.cx,
            map_cy=grids.cy,
            map_sx=grids.sx,
            map_sy=grids.sy,
            map_cu=np.ascontiguousarray(cu),
            map_cv=np.ascontiguousarray(cv),
        )
    return model


class NativeCoaddPsf(BasePsf):
    """Drop-in replacement for ``xlens`` LsstPsf backed by the native
    model: ``draw(x, y)`` returns the ``computeImage`` stamp resized to
    ``(npix, npix)`` with the same conventions.

    ``x``/``y`` are LOCAL (array) pixel coordinates when ``lsst_bbox``
    is given, exactly like ``LsstPsf``.
    """

    def __init__(self, model: CoaddPsfModel, npix: int, lsst_bbox=None):
        super().__init__()
        self.model = model
        self.npix = int(npix)
        self.shape = (self.npix, self.npix)
        if lsst_bbox is None:
            self.x_min = 0.0
            self.y_min = 0.0
        else:
            min_corner = lsst_bbox.getMin()
            self.x_min = min_corner.getX()
            self.y_min = min_corner.getY()

    @property
    def native_model(self):
        """The C++ PerSourcePsf handle (drawn inside ForceTask)."""
        return self.model

    def draw(self, x, y) -> NDArray:
        return self.model.draw(
            float(x) + self.x_min, float(y) + self.y_min, self.npix
        )
