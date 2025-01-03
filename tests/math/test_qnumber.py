import anacal
import numpy as np


def test_qnumber():
    q1 = anacal.math.qnumber(1.0, 2.0, 3.0, 4.0, 5.0)
    q2 = anacal.math.array_to_qnumber(np.array([2.0, 3.0, 4.0, 5.0, 6.0]))

    np.testing.assert_almost_equal(
        anacal.math.qnumber_to_array(q1 + q2),
        np.array([3.0, 5.0, 7.0, 9.0, 11.0]),
    )

    np.testing.assert_almost_equal(
        anacal.math.qnumber_to_array(q1 * 3.0),
        np.array([3.0, 6.0, 9.0, 12.0, 15.0]),
    )

    np.testing.assert_almost_equal(
        anacal.math.qnumber_to_array(3.0 * q1),
        np.array([3.0, 6.0, 9.0, 12.0, 15.0]),
    )

    np.testing.assert_almost_equal(
        anacal.math.qnumber_to_array(q1 / 2.0),
        np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
    )

    np.testing.assert_almost_equal(
        anacal.math.qnumber_to_array(q1 * q2),
        np.array([2.0, 7.0, 10.0, 13.0, 16.0]),
    )

    np.testing.assert_almost_equal(
        anacal.math.qnumber_to_array(q1 / q2),
        np.array([0.5, 0.25, 0.5, 3.0 / 4.0, 1.0]),
    )

    q1 = anacal.math.qnumber(1.0, 2.0, 3.0, 4.0, 5.0)
    e = np.exp(1.0)
    e2 = np.exp(2.0)
    np.testing.assert_almost_equal(
        anacal.math.qnumber_to_array(anacal.math.exp(q1)),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * e,
    )
    np.testing.assert_almost_equal(
        anacal.math.qnumber_to_array(anacal.math.exp(q2)),
        np.array([1.0, 3.0, 4.0, 5.0, 6.0]) * e2,
    )

    return
