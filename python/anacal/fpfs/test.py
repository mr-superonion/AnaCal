import math
import numpy as np


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
    v = np.piecewise(
        t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _ssfunc2, 1.0]
    )
    return v


def ssfunc2_deriv(x, mu, sigma):
    """Returns the C2 smooth step weight funciton

    Args:
    x (NDArray): input data vector
    mu (float): center of the cut
    sigma (float): half width of the selection function

    Returns:
    out (NDArray): the weight funciton
    """

    t = (x - mu) / sigma / 2.0 + 0.5
    dv = (
        np.piecewise(
            t, [t < 0, (t >= 0) & (t <= 1), t > 1], [0.0, _ssfunc2_deriv, 0.0]
        )
        / sigma
        / 2.0
    )
    return dv


def _dg(self, x):
    m00_g1 = -math.sqrt(2.0) * (x["m22c"])
    m00_g2 = -math.sqrt(2.0) * (x["m22s"])
    m20_g1 = -math.sqrt(6.0) * (x["m42c"])
    m20_g2 = -math.sqrt(6.0) * (x["m42s"])
    m22c_g1 = (
        1.0 / math.sqrt(2.0) * (x["m00"] - x["m40"])
        - math.sqrt(3.0) * x["m44c"]
    )
    m22s_g2 = (
        1.0 / math.sqrt(2.0) * (x["m00"] - x["m40"])
        + math.sqrt(3.0) * x["m44c"]
    )
    # off diagonal term
    # m22c_g2 = (
    #     - math.sqrt(3.0) * x["m44s"]
    # )

    # m22s_g1 = (
    #     - math.sqrt(3.0) * x["m44s"]
    # )
    m42c_g1 = (
        math.sqrt(6.0) / 2.0 * (x["m20"] - x["m60"])
        - math.sqrt(5.0) * x["m64c"]
    )
    m42s_g2 = (
        math.sqrt(6.0) / 2.0 * (x["m20"] - x["m60"])
        + math.sqrt(5.0) * x["m64c"]
    )
    # off diagonal term
    # m22c_g2 = (
    #     - math.sqrt(5.0) * x["m64s"]
    # )

    # m22s_g1 = (
    #     - math.sqrt(5.0) * x["m64s"]
    # )
    return (m00_g1, m00_g2, m20_g1, m20_g2, m22c_g1, m22s_g2, m42c_g1, m42s_g2)


def _ell(self, x, x_dg, C0):
    _denom = x["m00"] + C0
    # ellipticity1
    e1 = x["m22c"] / _denom
    e1_g1 = x_dg["m22c_g1"] / _denom - x_dg["m00_g1"] * x["m22c"] / (_denom) ** 2.0

    # ellipticity2
    e2 = x["m22s"] / _denom
    e2_g2 = x_dg["m22s_g2"] / _denom - x_dg["m00_g2"] * x["m22s"] / (_denom) ** 2.0

    # ellipticity1 (4th order)
    q1 = x["m42c"] / _denom
    q1_g1 = x_dg["m42c_g1"] / _denom - x_dg["m00_g1"] * x["m42c"] / (_denom) ** 2.0

    # ellipticity2 (4th order)
    q2 = x["m42s"] / _denom
    q2_g2 = x_dg["m42s_g2"] / _denom - x_dg["m00_g2"] * x["m42s"] / (_denom) ** 2.0
    return e1, e1_g1, e2, e2_g2


def _wsel(self, x, m00_g1, m00_g2, m20_g1, m20_g2):
    # selection on flux
    w0l = ssfunc2(x["m00"], self.m00_min, self.sigma_m00)
    dw0l = ssfunc2_deriv(x["m00"], self.m00_min, self.sigma_m00)
    w0l_g1 = dw0l * m00_g1
    w0l_g2 = dw0l * m00_g2

    w0u = ssfunc2(-x["m00"], -500, self.sigma_m00)
    dw0u = ssfunc2_deriv(-x["m00"], -500, self.sigma_m00)
    w0u_g1 = dw0u * -m00_g1
    w0u_g2 = dw0u * -m00_g2

    # selection on size (lower limit)
    # (M00 + M20) / M00 > r2_min
    r2l = x["m00"] * (1.0 - self.r2_min) + x["m20"]
    w2l = ssfunc2(
        r2l,
        self.sigma_r2,
        self.sigma_r2,
    )
    dw2l = ssfunc2_deriv(
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
    det0 = ssfunc2(
        x["v0"],
        self.sigma_v - self.pcut,
        self.sigma_v,
    )
    det0_deriv = ssfunc2_deriv(
        x["v0"],
        self.sigma_v - self.pcut,
        self.sigma_v,
    )
    det0_g1 = det0_deriv * (x - 2.0 * y)["v0r1"]
    det0_g2 = det0_deriv * (x - 2.0 * y)["v0r2"]

    det1 = ssfunc2(
        x["v1"],
        self.sigma_v - self.pcut,
        self.sigma_v,
    )
    det1_deriv = ssfunc2_deriv(
        x["v1"],
        self.sigma_v - self.pcut,
        self.sigma_v,
    )
    det1_g1 = det1_deriv * (x - 2.0 * y)["v1r1"]
    det1_g2 = det1_deriv * (x - 2.0 * y)["v1r2"]

    det2 = ssfunc2(
        x["v2"],
        self.sigma_v - self.pcut,
        self.sigma_v,
    )
    det2_deriv = ssfunc2_deriv(
        x["v2"],
        self.sigma_v - self.pcut,
        self.sigma_v,
    )
    det2_g1 = det2_deriv * (x - 2.0 * y)["v2r1"]
    det2_g2 = det2_deriv * (x - 2.0 * y)["v2r2"]

    det3 = ssfunc2(
        x["v3"],
        self.sigma_v - self.pcut,
        self.sigma_v,
    )
    det3_deriv = ssfunc2_deriv(
        x["v3"],
        self.sigma_v - self.pcut,
        self.sigma_v,
    )
    det3_g1 = det3_deriv * (x - 2.0 * y)["v3r1"]
    det3_g2 = det3_deriv * (x - 2.0 * y)["v3r2"]

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
    wdet = ssfunc2(w, self.pthres, fpfs_det_sigma2)
    wdet_deriv = ssfunc2_deriv(w, self.pthres, fpfs_det_sigma2)
    return wdet, wdet_deriv * w_g1, wdet_deriv * w_g2


def _run(self, x, y):
    m00_g1, m00_g2, m20_g1, m20_g2, m22c_g1, m22s_g2, m42c_g1, m42s_g2 \
        = self._dg(x - 2.0 * y)
    e1, e1_g1, e2, e2_g2 = self._ell(
        x,
        m00_g1,
        m00_g2,
        m22c_g1,
        m22s_g2,
    )
    out = [e1, e1_g1, e2, e2_g2]
    q1, q1_g1, q2, q2_g2 = self._ell_4th(
        x,
        m00_g1,
        m00_g2,
        m42c_g1,
        m42s_g2,
    )
    out.extend([q1, q1_g1, q2, q2_g2])
    wsel, wsel_g1, wsel_g2 = self._wsel(
        x, m00_g1, m00_g2, m20_g1, m20_g2
    )
    wdet, wdet_g1, wdet_g2 = self._wdet(x, y)
    w = wdet * wsel
    w = wdet * wsel
    w_g1 = wdet_g1 * wsel + wdet * wsel_g1
    w_g2 = wdet_g2 * wsel + wdet * wsel_g2
    flux = (
        (x["m00"] + x["m20"])
        * (self.sigma_arcsec / self.pixel_scale) ** 2.0
        / 2.0
    )
    out.extend([w, w_g1, w_g2, flux])
    return np.array(out)
