# FPFS shear estimator
# Copyright 20210805 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# python lib
import jax
import jax.numpy as jnp
import numpy as np
import numpy.lib.recfunctions as rfn

from . import fpfs_cut_sigma_ratio, fpfs_det_sigma2
from .base import get_det_col_names, get_shapelets_col_names
from .table import FpfsCatalog, FpfsCovariance

snr_min_default = 12.0
r2_min_default = 0.05
r2_max_default = 2.0
c0_default = 2.55


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


class CatIaskBase(object):
    def __init__(self, nord=-1, det_nrot=-1):
        self.nord = nord
        self.det_nrot = det_nrot
        self.dtype = None
        return

    def _run(self, x: jnp.float64, y: jnp.float64 = 0.0):
        return jnp.array([0])

    def run(self, cat: FpfsCatalog):
        """This function meausres observables and corresponding shear response

        Args:
        cat (FpfsCatalog): Input source catalog

        Returns:
        result (NDArray):   Measurements
        """
        assert cat.nord == self.nord, "input has wrong nord"
        assert cat.det_nrot == self.det_nrot, "input has wrong det_nrot"
        if cat.noise is None:
            func = jax.vmap(
                self._run,
                in_axes=0,
                out_axes=0,
            )
            result = func(cat.array)
        else:
            assert cat.noise is not None
            func = jax.vmap(
                self._run,
                in_axes=(0, 0),
                out_axes=0,
            )
            result = func(cat.array, cat.noise)

        result = np.core.records.fromarrays(
            result.transpose(),
            dtype=self.dtype,
        )
        return result


class CatTaskS(CatIaskBase):
    def __init__(
        self,
        nord: int,
        pixel_scale: float,
        sigma_arcsec: float,
        mag_zero: float,
        cov_matrix: FpfsCovariance,
    ):
        super().__init__(nord=nord)
        assert nord == 4
        self.pixel_scale = pixel_scale
        self.sigma_arcsec = sigma_arcsec
        self.mag_zero = mag_zero
        name_s, _ = get_shapelets_col_names(nord)
        self.di = {element: index for index, element in enumerate(name_s)}

        self.ncol = len(name_s)
        self.colnames = name_s
        self.dtype = [
            ("e1", "f8"),
            ("e1_g1", "f8"),
            ("e2", "f8"),
            ("e2_g2", "f8"),
            ("flux", "f8"),
        ]
        if not set(self.colnames).issubset(cov_matrix.colnames):
            raise ValueError(
                "Input covariance matrix of shapelets has a wrong colnames"
            )
        if not cov_matrix.nord == nord:
            raise ValueError(
                "Input covariance matrix of shapelets has wrong nord"
            )

        # standard deviation
        self.cov_matrix = cov_matrix
        std_m00 = cov_matrix.std_m00
        std_r2 = cov_matrix.std_r2
        # control steepness of the selection function
        self.sigma_m00 = fpfs_cut_sigma_ratio * std_m00
        self.sigma_r2 = fpfs_cut_sigma_ratio * std_r2
        self.snr_min = snr_min_default
        self.m00_min = snr_min_default * std_m00
        self.r2_min = r2_min_default
        self.r2_max = r2_max_default
        # shape parameters
        self.C0 = c0_default * std_m00
        return

    def update_parameters(
        self,
        snr_min: float | None = None,
        r2_min: float | None = None,
        r2_max: float | None = None,
        c0: float | None = None,
    ):
        if snr_min is not None:
            # selection on SNR
            self.snr_min = snr_min
            self.m00_min = snr_min * self.cov_matrix.std_m00
        if r2_min is not None:
            self.r2_min = r2_min
        if r2_max is not None:
            self.r2_max = r2_max
        if c0 is not None:
            # shape parameters
            self.C0 = c0 * self.cov_matrix.std_m00
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

        # m22c_g2 = (
        #     - jnp.sqrt(3.0) * x[self.di["m44s"]]
        # )

        # m22s_g1 = (
        #     - jnp.sqrt(3.0) * x[self.di["m44s"]]
        # )
        m22s_g2 = (
            1.0 / jnp.sqrt(2.0) * (x[self.di["m00"]] - x[self.di["m40"]])
            + jnp.sqrt(3.0) * x[self.di["m44c"]]
        )
        return m00_g1, m00_g2, m20_g1, m20_g2, m22c_g1, m22s_g2

    def _run(self, x, y=0.0):
        m00_g1, m00_g2, m20_g1, m20_g2, m22c_g1, m22s_g2 = self._dg(x - 2.0 * y)

        _denom = x[self.di["m00"]] + self.C0
        # ellipticity1
        e1 = x[self.di["m22c"]] / _denom
        # ellipticity2
        e2 = x[self.di["m22s"]] / _denom

        e1_g1 = m22c_g1 / _denom - m00_g1 * x[self.di["m22c"]] / (_denom) ** 2.0
        e2_g2 = m22s_g2 / _denom - m00_g2 * x[self.di["m22s"]] / (_denom) ** 2.0

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

        we1 = wsel * e1
        we2 = wsel * e2
        we1_g1 = wsel_g1 * e1 + wsel * e1_g1
        we2_g2 = wsel_g2 * e2 + wsel * e2_g2
        flux = (
            (x[self.di["m00"]] + x[self.di["m20"]])
            * (self.sigma_arcsec / self.pixel_scale) ** 2.0
            / 2.0
        )
        return jnp.array([we1, we1_g1, we2, we2_g2, flux])


class CatTaskD(CatIaskBase):
    def __init__(
        self,
        det_nrot: int,
        sigma_arcsec_det: float,
        cov_matrix: FpfsCovariance,
        pthres: float = 0.8,
        pthres2: float = 0.12,
    ):
        super().__init__(det_nrot=det_nrot)
        assert det_nrot == 4
        self.det_nrot = det_nrot
        self.sigma_arcsec_det = sigma_arcsec_det
        name_d = get_det_col_names(det_nrot)
        self.di = {element: index for index, element in enumerate(name_d)}
        self.ncol = len(name_d)
        self.colnames = name_d
        self.dtype = [
            ("wdet", "f8"),
            ("wdet_g1", "f8"),
            ("wdet_g2", "f8"),
        ]
        if not set(self.colnames).issubset(cov_matrix.colnames):
            raise ValueError(
                "Input covariance matrix of detection has a wrong colnames"
            )
        if not cov_matrix.det_nrot == det_nrot:
            raise ValueError(
                "Input covariance matrix of detection has wrong det_nrot"
            )
        self.cov_matrix = cov_matrix
        std_v = cov_matrix.std_v
        # control steepness
        self.sigma_v = fpfs_cut_sigma_ratio * std_v
        # detection threshold
        self.pcut = pthres * std_v
        self.pthres2 = pthres2
        return

    def _run(self, x, y=0.0):
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
        wdet, wdet_deriv = ssfunc2(w, self.pthres2, 0.04)
        return jnp.array([wdet, wdet_deriv * w_g1, wdet_deriv * w_g2])


class CatalogTask:
    def __init__(
        self,
        nord: int,
        det_nrot: int,
        pixel_scale: float,
        sigma_arcsec: float,
        mag_zero: float,
        cov_matrix_s: FpfsCovariance,
        cov_matrix_d: FpfsCovariance,
        sigma_arcsec_det: float | None = None,
        pthres: float = 0.8,
        pthres2: float = 0.12,
    ):
        """Fpfs catalog task"""
        self.shapelet_task = CatTaskS(
            nord=nord,
            pixel_scale=pixel_scale,
            sigma_arcsec=sigma_arcsec,
            mag_zero=mag_zero,
            cov_matrix=cov_matrix_s,
        )
        if sigma_arcsec_det is None:
            sigma_arcsec_det = sigma_arcsec
        self.det_task = CatTaskD(
            det_nrot=det_nrot,
            sigma_arcsec_det=sigma_arcsec_det,
            cov_matrix=cov_matrix_d,
            pthres=pthres,
            pthres2=pthres2,
        )
        self.dtype = [
            ("e1", "f8"),
            ("e1_g1", "f8"),
            ("e2", "f8"),
            ("e2_g2", "f8"),
            ("flux", "f8"),
            ("wdet", "f8"),
            ("wdet_g1", "f8"),
            ("wdet_g2", "f8"),
            ("mask", "?"),
        ]
        return

    def update_parameters(
        self,
        snr_min: float | None = None,
        r2_min: float | None = None,
        r2_max: float | None = None,
        c0: float | None = None,
    ):
        self.shapelet_task.update_parameters(
            snr_min=snr_min,
            r2_min=r2_min,
            r2_max=r2_max,
            c0=c0,
        )

    def run(
        self,
        shapelet: FpfsCatalog,
        detection: FpfsCatalog,
    ):
        """This function returns the shape and shear response of shape using
        shapelet catalog and detection catalog

        Args:
        shapelet (FpfsCatalog): shapelet catalog
        detection (FpfsCatalog): detection catalog

        Returns:
        src (NDArray): shape catalog
        """
        assert shapelet.nord == self.shapelet_task.nord
        assert detection.det_nrot == self.det_task.det_nrot
        array_s = self.shapelet_task.run(
            cat=shapelet,
        )

        array_d = self.det_task.run(
            cat=detection,
        )
        # mask out those with negligible weight
        mask = (
            array_d["wdet"]
            * np.sqrt(array_s["e1"] ** 2.0 + array_s["e2"] ** 2.0)
        ) > 1e-10
        array_m = np.array(mask, dtype=[("mask", "?")])
        src = rfn.merge_arrays(
            (array_s, array_d, array_m),
            flatten=True,
            usemask=False,
        )
        return src


def m2e(mm, const=1.0, nn=None):
    """Estimates FPFS ellipticities from fpfs moments

    Args:
    mm (NDArray): FPFS moments
    const (float): the weight constant [default:1]
    nn (NDArray): noise covaraince elements [default: None]

    Returns:
    out (NDArray):
        an array of [FPFS ellipticities, FPFS ellipticity response, FPFS flux,
        size and FPFS selection response]
    """

    # ellipticity, q-ellipticity, sizes, e^2, eq
    types = [
        ("fpfs_e1", "<f8"),
        ("fpfs_e2", "<f8"),
        ("fpfs_R1E", "<f8"),
        ("fpfs_R2E", "<f8"),
    ]
    # make the output NDArray
    out = np.array(np.zeros(mm.size), dtype=types)

    # FPFS shape weight's inverse
    _w = mm["fpfs_m00"] + const
    # FPFS ellipticity
    e1 = mm["fpfs_m22c"] / _w
    e2 = mm["fpfs_m22s"] / _w
    # FPFS spin-0 observables
    s0 = mm["fpfs_m00"] / _w
    s4 = mm["fpfs_m40"] / _w
    # intrinsic ellipticity
    e1e1 = e1 * e1
    e2e2 = e2 * e2

    # spin-2 properties
    out["fpfs_e1"] = e1  # ellipticity
    out["fpfs_e2"] = e2
    # response for ellipticity
    out["fpfs_R1E"] = (s0 - s4 + 2.0 * e1e1) / np.sqrt(2.0)
    out["fpfs_R2E"] = (s0 - s4 + 2.0 * e2e2) / np.sqrt(2.0)
    return out
