import anacal
import numpy as np
import pytest

# 3..7 are the common piff choices; 11 is what DP1 uses.
LANCZOS_ORDERS = (3, 4, 5, 6, 7, 9, 11)

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


class my_psf(anacal.psf.BasePsf):

    def __init__(self, a):
        super().__init__()
        self.a = a

    def draw(self, x, y):
        return np.ones((10, 10))


def test_pypsf():
    psf_obj = my_psf(a=1)
    del psf_obj
    return


def test_lanczos_kvals_are_the_raw_kernel_transform():
    """The Gauss-Legendre K_m equal a brute-force integral."""
    for n in LANCZOS_ORDERS:
        kv = anacal.psf._lanczos_dc_kvals(n)
        assert kv.shape == (5,)
        x = np.linspace(-n, n, 2_000_001)
        raw = np.sinc(x) * np.sinc(x / n)
        brute = np.array(
            [
                _trapezoid(raw * np.cos(2.0 * np.pi * m * x), x)
                for m in range(1, 6)
            ]
        )
        np.testing.assert_allclose(kv, brute, rtol=0, atol=1e-12)
        # the aliasing terms fall off with frequency
        assert np.all(np.abs(kv[1:]) < np.abs(kv[:-1]))


def test_lanczos_kernel_interpolates_and_conserves_dc():
    """What the correction is for, checked without galsim.

    The kernel must still interpolate (1 at 0, 0 at other integers,
    0 beyond n), an empty K must give the raw kernel back, and the sum
    over integer shifts -- the response to a flat field -- must be
    much closer to 1 with the correction than without.
    """
    for n in LANCZOS_ORDERS:
        kv = anacal.psf._lanczos_dc_kvals(n)
        none = np.zeros(0)

        k = np.arange(-n - 2, n + 3, dtype=float)
        np.testing.assert_allclose(
            anacal.psf.lanczos_kernel(k, n, kv),
            (k == 0).astype(float),
            rtol=0,
            atol=1e-15,
        )

        x = np.linspace(-n + 0.01, n - 0.01, 1001)
        np.testing.assert_allclose(
            anacal.psf.lanczos_kernel(x, n, none),
            np.sinc(x) * np.sinc(x / n),
            rtol=0,
            atol=1e-15,
        )

        frac = np.linspace(0.0, 1.0, 101, endpoint=False)
        grid = frac[None, :] + np.arange(-n, n + 1)[:, None]
        err_dc = np.max(
            np.abs(anacal.psf.lanczos_kernel(grid, n, kv).sum(axis=0) - 1)
        )
        err_raw = np.max(
            np.abs(anacal.psf.lanczos_kernel(grid, n, none).sum(axis=0) - 1)
        )
        assert err_dc < 1e-5
        assert err_dc < err_raw / 10.0


def test_lanczos_kernel_matches_galsim():
    """PIFF models are fit with galsim's kernel, so match it, not the
    ideal: galsim keeps five aliasing terms and so do we."""
    galsim = pytest.importorskip("galsim")
    for n in LANCZOS_ORDERS:
        x = np.linspace(-n + 0.013, n - 0.021, 1777)
        for conserve_dc in (False, True):
            interp = galsim.Lanczos(n, conserve_dc=conserve_dc)
            ref = np.array([interp.xval(float(v)) for v in x])
            kv = anacal.psf._lanczos_dc_kvals(n) if conserve_dc else None
            got = anacal.psf.lanczos_kernel(
                x, n, kv if kv is not None else np.zeros(0)
            )
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-13)


if __name__ == "__main__":
    test_pypsf()
