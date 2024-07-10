# Catalog Tasks of Shapelet Based Measurements
#
# python lib
import jax
import jax.numpy as jnp
import numpy as np
import numpy.lib.recfunctions as rfn

from . import fpfs_cut_sigma_ratio, fpfs_det_sigma2, fpfs_pnr
from .base import get_det_col_names, get_shapelets_col_names
from .table import Catalog, Covariance

snr_min_default = 12.0
r2_min_default = 0.1
r2_max_default = 2.0
c0_default = 3.0
pthres_default = 0.12


def _ssfunc1(t):
    return -2.0 * t**3.0 + 3 * t**2.0


def _ssfunc1_deriv(t):
    return -6.0 * t**2.0 + 6 * t**1.0


def ssfunc1(x, mu, sigma):
    """Returns the C1 smooth step weight funciton

    Args:
    x (NDArray): input data vector
    mu (float): center of the cut
    sigma (float): half width of the selection function

    Returns:
    out (NDArray): the weight funciton
    """

    t = (x - mu) / sigma / 2.0 + 0.5
    v = jnp.piecewise(
        t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _ssfunc1, 1.0]
    )
    dv = (
        jnp.piecewise(
            t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _ssfunc1_deriv, 0.0]
        )
        / sigma
        / 2.0
    )
    return v, dv


def _ssfunc2(t):
    return 6 * t**5.0 - 15 * t**4.0 + 10 * t**3.0


def _ssfunc2_deriv(t):
    return 30 * t**4.0 - 60 * t**3.0 + 30 * t**2.0


def ssfunc2(x, mu, sigma):
    """Returns the C2 smooth step weight funciton

    Args:
    x (NDArray): input data vector
    mu (float): center of the cut
    sigma (float): half width of the selection function

    Returns:
    out (NDArray): the weight funciton
    """

    t = (x - mu) / sigma / 2.0 + 0.5
    v = jnp.piecewise(
        t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _ssfunc2, 1.0]
    )
    dv = (
        jnp.piecewise(
            t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _ssfunc2_deriv, 0.0]
        )
        / sigma
        / 2.0
    )
    return v, dv


class CatTaskBase(object):
    def __init__(
        self,
        cov_matrix: Covariance,
        nord: int = -1,
        det_nrot: int = -1,
    ):
        self.nord = nord
        self.det_nrot = det_nrot
        self.colnames = []
        self.cov_matrix = cov_matrix
        self.dtype = []

        # Initialize the column names
        # and define cuts dependends on covariance matrix
        if self.nord >= 4:
            if not hasattr(cov_matrix, "std_m00"):
                raise ValueError(
                    "The input covariance does not have std_m00",
                    "which is required for nord = %d" % self.nord,
                )
            if not hasattr(cov_matrix, "std_r2"):
                raise ValueError(
                    "The input covariance does not have std_r2",
                    "which is required for nord = %d" % self.nord,
                )
            # Initialize column names
            snames, _ = get_shapelets_col_names(nord)
            self.colnames = self.colnames + snames
            # standard deviation
            std_m00 = cov_matrix.std_m00
            std_r2 = cov_matrix.std_r2
            # control steepness of the selection function
            # (do not change them)
            self.sigma_m00 = fpfs_cut_sigma_ratio * std_m00
            self.sigma_r2 = fpfs_cut_sigma_ratio * std_r2
            # (can be updated)
            self.snr_min = snr_min_default
            self.m00_min = self.snr_min * std_m00
            self.r2_min = r2_min_default
            self.r2_max = r2_max_default
            self.C0 = c0_default * std_m00

            self.dtype.extend(
                [
                    ("e1", "f8"),  # shape
                    ("e1_g1", "f8"),  # shear response of shape
                    ("e2", "f8"),  # shape
                    ("e2_g2", "f8"),  # shear response of shape
                ]
            )
            if self.nord >= 6:
                self.dtype.extend(
                    [
                        ("q1", "f8"),  # shape (4th order)
                        ("q1_g1", "f8"),  # shear response of shape
                        ("q2", "f8"),  # shape (4th order)
                        ("q2_g2", "f8"),  # shear response of shape
                    ]
                )

        if self.det_nrot >= 4:
            if not hasattr(cov_matrix, "std_v"):
                raise ValueError(
                    "The input covariance does not have std_r2",
                    "which is required for det_nrot = %d" % self.std_v,
                )
            dnames = get_det_col_names(det_nrot)
            self.colnames = self.colnames + dnames
            self.std_v = cov_matrix.std_v

            # control steepness
            self.sigma_v = fpfs_cut_sigma_ratio * self.std_v
            # (do not change it)
            self.pcut = fpfs_pnr * self.std_v
            # (can be updated)
            self.pthres = pthres_default
            self.dtype.extend(
                [
                    ("w", "f8"),  # weight (detection and selection)
                    ("w_g1", "f8"),  # shear response of weight
                    ("w_g2", "f8"),  # shear response of weight
                    ("flux", "f8"),  # flux
                ]
            )

        assert len(self.colnames) > 0
        self.di = {
            element: index for index, element in enumerate(self.colnames)
        }
        self.ncol = len(self.colnames)
        if not set(self.colnames).issubset(cov_matrix.colnames):
            raise ValueError(
                "Input covariance matrix has a different colnames from",
                "nord = %d, det_nrot = %d" % (self.nord, self.det_nrot),
            )
        return

    def update_parameters(
        self,
        snr_min: float | None = None,
        r2_min: float | None = None,
        r2_max: float | None = None,
        c0: float | None = None,
        pthres: float | None = None,
    ):
        # selection on SNR
        if snr_min is not None:
            if self.nord >= 4:
                self.snr_min = snr_min
                self.m00_min = snr_min * self.cov_matrix.std_m00
            else:
                raise RuntimeError("Cannot update srn_min")
        # selection on size
        if r2_min is not None:
            if self.nord >= 4:
                self.r2_min = r2_min
            else:
                raise RuntimeError("Cannot update r2_min")
        if r2_max is not None:
            if self.nord >= 4:
                self.r2_max = r2_max
            else:
                raise RuntimeError("Cannot update r2_min")
        # shape parameters
        if c0 is not None:
            if self.nord >= 4:
                self.C0 = c0 * self.cov_matrix.std_m00
            else:
                raise RuntimeError("Cannot update c0")
        # detection threshold
        if pthres is not None:
            if self.nord >= 4:
                self.pthres = pthres
            else:
                raise RuntimeError("Cannot update pthres")
        return

    def _dg(self, x):
        m00_g1 = -jnp.sqrt(2.0) * (x[self.di["m22c"]])
        m00_g2 = -jnp.sqrt(2.0) * (x[self.di["m22s"]])
        m20_g1 = -jnp.sqrt(6.0) * (x[self.di["m42c"]])
        m20_g2 = -jnp.sqrt(6.0) * (x[self.di["m42s"]])
        m22c_g1 = (
            1.0 / jnp.sqrt(2.0) * (x[self.di["m00"]] - x[self.di["m40"]])
            - jnp.sqrt(3.0) * x[self.di["m44c"]]
        )
        m22s_g2 = (
            1.0 / jnp.sqrt(2.0) * (x[self.di["m00"]] - x[self.di["m40"]])
            + jnp.sqrt(3.0) * x[self.di["m44c"]]
        )
        # off diagonal term
        # m22c_g2 = (
        #     - jnp.sqrt(3.0) * x[self.di["m44s"]]
        # )

        # m22s_g1 = (
        #     - jnp.sqrt(3.0) * x[self.di["m44s"]]
        # )
        return (m00_g1, m00_g2, m20_g1, m20_g2, m22c_g1, m22s_g2)

    def _dg_4th(self, x):
        m42c_g1 = (
            jnp.sqrt(6.0) / 2.0 * (x[self.di["m20"]] - x[self.di["m60"]])
            - jnp.sqrt(5.0) * x[self.di["m64c"]]
        )
        m42s_g2 = (
            jnp.sqrt(6.0) / 2.0 * (x[self.di["m20"]] - x[self.di["m60"]])
            + jnp.sqrt(5.0) * x[self.di["m64c"]]
        )
        # off diagonal term
        # m22c_g2 = (
        #     - jnp.sqrt(5.0) * x[self.di["m64s"]]
        # )

        # m22s_g1 = (
        #     - jnp.sqrt(5.0) * x[self.di["m64s"]]
        # )
        return (m42c_g1, m42s_g2)

    def _ell(self, x, m00_g1, m00_g2, m22c_g1, m22s_g2):
        _denom = x[self.di["m00"]] + self.C0
        # ellipticity1
        e1 = x[self.di["m22c"]] / _denom
        e1_g1 = m22c_g1 / _denom - m00_g1 * x[self.di["m22c"]] / (_denom) ** 2.0

        # ellipticity2
        e2 = x[self.di["m22s"]] / _denom
        e2_g2 = m22s_g2 / _denom - m00_g2 * x[self.di["m22s"]] / (_denom) ** 2.0
        return e1, e1_g1, e2, e2_g2

    def _ell_4th(self, x, m00_g1, m00_g2, m42c_g1, m42s_g2):
        _denom = x[self.di["m00"]] + self.C0
        # ellipticity1 (4th order)
        q1 = x[self.di["m42c"]] / _denom
        q1_g1 = m42c_g1 / _denom - m00_g1 * x[self.di["m42c"]] / (_denom) ** 2.0

        # ellipticity2 (4th order)
        q2 = x[self.di["m42s"]] / _denom
        q2_g2 = m42s_g2 / _denom - m00_g2 * x[self.di["m42s"]] / (_denom) ** 2.0
        return q1, q1_g1, q2, q2_g2

    def _wsel(self, x, m00_g1, m00_g2, m20_g1, m20_g2):
        # selection on flux
        w0l, dw0l = ssfunc2(x[self.di["m00"]], self.m00_min, self.sigma_m00)
        w0l_g1 = dw0l * m00_g1
        w0l_g2 = dw0l * m00_g2

        w0u, dw0u = ssfunc2(-x[self.di["m00"]], -500, self.sigma_m00)
        w0u_g1 = dw0u * -m00_g1
        w0u_g2 = dw0u * -m00_g2

        # selection on size (lower limit)
        # (M00 + M20) / M00 > r2_min
        r2l = x[self.di["m00"]] * (1.0 - self.r2_min) + x[self.di["m20"]]
        w2l, dw2l = ssfunc2(
            r2l,
            self.sigma_r2,
            self.sigma_r2,
        )
        w2l_g1 = dw2l * (m00_g1 * (1.0 - self.r2_min) + m20_g1)
        w2l_g2 = dw2l * (m00_g2 * (1.0 - self.r2_min) + m20_g2)

        wsel = w0l * w0u * w2l
        wsel_g1 = w0l_g1 * w0u * w2l + w0l * w0u_g1 * w2l + w0l * w0u * w2l_g1
        wsel_g2 = w0l_g2 * w0u * w2l + w0l * w0u_g2 * w2l + w0l * w0u * w2l_g2
        return wsel, wsel_g1, wsel_g2

    def _wdet(self, x, y):
        det0, det0_deriv = ssfunc2(
            x[self.di["v0"]],
            self.sigma_v - self.pcut,
            self.sigma_v,
        )
        det0_g1 = det0_deriv * (x - 2.0 * y)[self.di["v0r1"]]
        det0_g2 = det0_deriv * (x - 2.0 * y)[self.di["v0r2"]]

        det1, det1_deriv = ssfunc2(
            x[self.di["v1"]],
            self.sigma_v - self.pcut,
            self.sigma_v,
        )
        det1_g1 = det1_deriv * (x - 2.0 * y)[self.di["v1r1"]]
        det1_g2 = det1_deriv * (x - 2.0 * y)[self.di["v1r2"]]

        det2, det2_deriv = ssfunc2(
            x[self.di["v2"]],
            self.sigma_v - self.pcut,
            self.sigma_v,
        )
        det2_g1 = det2_deriv * (x - 2.0 * y)[self.di["v2r1"]]
        det2_g2 = det2_deriv * (x - 2.0 * y)[self.di["v2r2"]]

        det3, det3_deriv = ssfunc2(
            x[self.di["v3"]],
            self.sigma_v - self.pcut,
            self.sigma_v,
        )
        det3_g1 = det3_deriv * (x - 2.0 * y)[self.di["v3r1"]]
        det3_g2 = det3_deriv * (x - 2.0 * y)[self.di["v3r2"]]

        w = det0 * det1 * det2 * det3
        w_g1 = (
            det0_g1 * det1 * det2 * det3
            + det0 * det1_g1 * det2 * det3
            + det0 * det1 * det2_g1 * det3
            + det0 * det1 * det2 * det3_g1
        )
        w_g2 = (
            det0_g2 * det1 * det2 * det3
            + det0 * det1_g2 * det2 * det3
            + det0 * det1 * det2_g2 * det3
            + det0 * det1 * det2 * det3_g2
        )
        wdet, wdet_deriv = ssfunc2(w, self.pthres, fpfs_det_sigma2)
        return wdet, wdet_deriv * w_g1, wdet_deriv * w_g2

    def _run(self, x, y):
        m00_g1, m00_g2, m20_g1, m20_g2, m22c_g1, m22s_g2 = self._dg(x - 2.0 * y)
        e1, e1_g1, e2, e2_g2 = self._ell(
            x,
            m00_g1,
            m00_g2,
            m22c_g1,
            m22s_g2,
        )
        out = [e1, e1_g1, e2, e2_g2]
        if self.nord >= 6:
            m42c_g1, m42s_g2 = self._dg_4th(x - 2.0 * y)
            q1, q1_g1, q2, q2_g2 = self._ell_4th(
                x,
                m00_g1,
                m00_g2,
                m42c_g1,
                m42s_g2,
            )
            out.extend([q1, q1_g1, q2, q2_g2])
        if self.det_nrot >= 4:
            wsel, wsel_g1, wsel_g2 = self._wsel(
                x, m00_g1, m00_g2, m20_g1, m20_g2
            )
            wdet, wdet_g1, wdet_g2 = self._wdet(x, y)
            w = wdet * wsel
            w = wdet * wsel
            w_g1 = wdet_g1 * wsel + wdet * wsel_g1
            w_g2 = wdet_g2 * wsel + wdet * wsel_g2
            flux = (
                (x[self.di["m00"]] + x[self.di["m20"]])
                * (self.sigma_arcsec / self.pixel_scale) ** 2.0
                / 2.0
            )
            out.extend([w, w_g1, w_g2, flux])
        return jnp.array(out)

    def _run_nn(self, x):
        return self._run(x, 0.0)

    def run(self, catalog: Catalog):
        """This function meausres observables and corresponding shear response

        Args:
        catalog (Catalog): Input source observable catalog

        Returns:
        result (NDArray):   Measurements
        """
        assert catalog.nord == self.nord, "input has wrong nord"
        assert catalog.det_nrot == self.det_nrot, "input has wrong det_nrot"
        self.pixel_scale = catalog.pixel_scale
        self.sigma_arcsec = catalog.sigma_arcsec
        self.mag_zero = catalog.mag_zero
        if catalog.noise is None:
            func = jax.vmap(
                self._run_nn,
                in_axes=0,
                out_axes=0,
            )
            result = func(catalog.array)
        else:
            assert catalog.noise is not None
            func = jax.vmap(
                self._run,
                in_axes=(0, 0),
                out_axes=0,
            )
            result = func(catalog.array, catalog.noise)

        result = np.core.records.fromarrays(
            result.transpose(),
            dtype=self.dtype,
        )
        return result


class CatalogTask:
    def __init__(
        self,
        nord: int,
        det_nrot: int,
        cov_matrix: Covariance,
    ):
        """Fpfs Catalog Task"""
        self.nord = nord
        self.det_nrot = det_nrot
        self.det_task = None
        self.meas_task = None
        if self.det_nrot >= 4 and self.nord >= 4:
            self.det_task = CatTaskBase(
                cov_matrix=cov_matrix,
                nord=nord,
                det_nrot=det_nrot,
            )
        if self.nord >= 4:
            self.meas_task = CatTaskBase(
                cov_matrix=cov_matrix,
                nord=nord,
                det_nrot=-1,
            )
            ndt = [(name + "_2", dtype) for name, dtype in self.meas_task.dtype]
            self.meas_task.dtype = ndt
        return

    def update_parameters(
        self,
        snr_min: float | None = None,
        r2_min: float | None = None,
        r2_max: float | None = None,
        c0: float | None = None,
        pthres: float | None = None,
    ):
        if self.det_task is not None:
            self.det_task.update_parameters(
                snr_min=snr_min,
                r2_min=r2_min,
                r2_max=r2_max,
                c0=c0,
                pthres=pthres,
            )

        if self.meas_task is not None:
            self.meas_task.update_parameters(
                c0=c0,
            )
        return

    def run(
        self,
        catalog: Catalog | None = None,
        catalog2: Catalog | None = None,
    ):
        """This function returns the shape and shear response of shape using
        shapelet catalog and detection catalog

        Args:
        catalog (Catalog | None): catalog with detection and shapelet
        catalog2 (Catalog | None): The secondary shapelet catalog

        Returns:
        src (NDArray): shape measurement array
        """
        src_list = []
        if catalog is not None:
            assert self.det_task is not None
            assert catalog.nord == self.det_task.nord
            assert catalog.det_nrot == self.det_task.det_nrot

            array_det = self.det_task.run(
                catalog=catalog,
            )
            src_list.append(array_det)

            # mask out those with negligible weight
            array_mask = np.array(
                (array_det["w"] > 1e-10),
                dtype=[("mask", "?")],
            )
            src_list.append(array_mask)

        if catalog2 is not None:
            assert self.meas_task is not None
            assert (
                catalog2.det_nrot == -1
            ), "The secondary catalog should not have detection columns"
            array_meas = self.meas_task.run(
                catalog=catalog2,
            )
            src_list.append(array_meas)

        if len(src_list) == 0:
            return
        else:
            return rfn.merge_arrays(
                src_list,
                flatten=True,
                usemask=False,
            )
