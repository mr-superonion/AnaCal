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


std_m00 = 1.0834360726086465


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
        )
    ],
    dtype=dtype,
)

c0 = 5.0


def test_catalog():
    dm_dg = anacal.fpfs.measure_shapelets_dg(mm_st, nn_st)
    ell = anacal.fpfs.measure_fpfs_shape(c0 * std_m00, mm_st, dm_dg)
    # measure_fpfs is the ellipticity path measure_shapelets_dg +
    # measure_fpfs_shape in one call; both must agree with the stored
    # fixed-point values.
    cat = anacal.fpfs.measure_fpfs(
        C0=c0 * std_m00,
        x_array=mm_st,
        y_array=nn_st,
    )

    for out in (ell, cat):
        np.testing.assert_array_almost_equal(out["e1"], e1)
        np.testing.assert_array_almost_equal(out["de1_dg1"], e1_g1)
        np.testing.assert_array_almost_equal(out["e2"], e2)
        np.testing.assert_array_almost_equal(out["de2_dg2"], e2_g2)
        np.testing.assert_array_almost_equal(out["q1"], q1)
        np.testing.assert_array_almost_equal(out["dq1_dg1"], q1_g1)
        np.testing.assert_array_almost_equal(out["q2"], q2)
        np.testing.assert_array_almost_equal(out["dq2_dg2"], q2_g2)
    return
