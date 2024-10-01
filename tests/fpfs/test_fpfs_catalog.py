import anacal
import numpy as np

e1 = 0.3995478
e1_g1 = 0.87003568
e2 = 0.17292894
e2_g2 = 1.03802606
q1 = 0.12069805
q1_g1 = -1.06407001
q2 = 0.14146077
q2_g2 = -0.2025206
w = 0.00526255
w_g1 = 2.81036601
w_g2 = 3.30901692
flux = 34.75230547


std_m00 = 1.0834360726086465
std_r2 = 2.5048658664072154
std_v = 1.251018909728324


dtype = [
    ("m00", "<f8"),
    ("m20", "<f8"),
    ("m22c", "<f8"),
    ("m22s", "<f8"),
    ("m40", "<f8"),
    ("m42c", "<f8"),
    ("m42s", "<f8"),
    ("m44c", "<f8"),
    ("m44s", "<f8"),
    ("m60", "<f8"),
    ("m64c", "<f8"),
    ("m64s", "<f8"),
    ("v0", "<f8"),
    ("v1", "<f8"),
    ("v2", "<f8"),
    ("v3", "<f8"),
    ("v0r1", "<f8"),
    ("v1r1", "<f8"),
    ("v2r1", "<f8"),
    ("v3r1", "<f8"),
    ("v0r2", "<f8"),
    ("v1r2", "<f8"),
    ("v2r2", "<f8"),
    ("v3r2", "<f8"),
]

mm_st = np.array(
    [
        (
            8.54511404,
            1.73663314,
            5.57860394,
            2.41448482,
            1.01227348,
            1.68522175,
            1.97511694,
            5.94473667,
            13.32679667,
            4.04315886,
            5.54036462,
            15.0743539,
            0.0942872,
            1.47942416,
            3.84008247,
            2.29085822,
            3.64885571,
            6.36896711,
            8.43807404,
            0.93694728,
            0.3926889,
            11.1870979,
            14.63340806,
            17.81840394,
        )
    ],
    dtype=dtype,
)
nn_st = np.array(
    [
        (
            4.98497322,
            3.27175016,
            -5.49274026,
            1.10021415,
            7.30996044,
            1.30472054,
            5.88266589,
            1.29045318,
            6.00610962,
            0.22930153,
            1.11395431,
            2.06584609,
            2.83092276,
            -6.01609028,
            -1.18313861,
            -1.15826497,
            -3.14368109,
            4.77983001,
            -2.06366764,
            1.86790905,
            -0.75719894,
            -7.11824815,
            -3.9632181,
            -1.77036605,
        )
    ],
    dtype=dtype,
)

snr_min = 8
r2_min = 0.1
c0 = 5.0
sigma_arcsec = 0.52
pixel_scale = 0.2
pthres = 0.12


def test_catalog():
    dm_dg = anacal.fpfs.measure_shapelets_dg(mm_st, nn_st)
    ell = anacal.fpfs.measure_fpfs_shape(c0 * std_m00, mm_st, dm_dg)
    cat = anacal.fpfs.measure_fpfs(
        C0=c0 * std_m00,
        std_v=std_v,
        pthres=pthres,
        m00_min=snr_min * std_m00,
        std_m00=std_m00,
        r2_min=r2_min,
        std_r2=std_r2,
        x_array=mm_st,
        y_array=nn_st,
    )

    np.testing.assert_array_almost_equal(
        ell["e1"],
        e1,
    )
    np.testing.assert_array_almost_equal(
        ell["e1_g1"],
        e1_g1,
    )
    np.testing.assert_array_almost_equal(
        ell["e2"],
        e2,
    )
    np.testing.assert_array_almost_equal(
        ell["e2_g2"],
        e2_g2,
    )
    np.testing.assert_array_almost_equal(
        ell["q1"],
        q1,
    )
    np.testing.assert_array_almost_equal(
        ell["q1_g1"],
        q1_g1,
    )
    np.testing.assert_array_almost_equal(
        ell["q2"],
        q2,
    )
    np.testing.assert_array_almost_equal(
        ell["q2_g2"],
        q2_g2,
    )

    np.testing.assert_array_almost_equal(
        cat["e1"],
        e1,
    )
    np.testing.assert_array_almost_equal(
        cat["e1_g1"],
        e1_g1,
    )
    np.testing.assert_array_almost_equal(
        cat["e2"],
        e2,
    )
    np.testing.assert_array_almost_equal(
        cat["e2_g2"],
        e2_g2,
    )
    np.testing.assert_array_almost_equal(
        cat["q1"],
        q1,
    )
    np.testing.assert_array_almost_equal(
        cat["q1_g1"],
        q1_g1,
    )
    np.testing.assert_array_almost_equal(
        cat["q2"],
        q2,
    )
    np.testing.assert_array_almost_equal(
        cat["q2_g2"],
        q2_g2,
    )

    np.testing.assert_array_almost_equal(
        cat["w"],
        w,
    )
    np.testing.assert_array_almost_equal(
        cat["w_g1"],
        w_g1,
    )
    np.testing.assert_array_almost_equal(
        cat["w_g2"],
        w_g2,
    )
    return
