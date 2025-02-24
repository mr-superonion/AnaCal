import anacal
import numpy as np


def test_tnumber():
    q1 = anacal.math.tnumber(1.0, 2.0, 3.0)
    q2 = anacal.math.tnumber([2.0, 3.0, 4.0])

    np.testing.assert_almost_equal(
        (-q1).to_array(),
        -np.array([1.0, 2.0, 3.0]),
    )

    np.testing.assert_almost_equal(
        (q1 + q2).to_array(),
        np.array([3.0, 5.0, 7.0]),
    )

    np.testing.assert_almost_equal(
        (q1 * 3.0).to_array(),
        np.array([3.0, 6.0, 9.0]),
    )

    np.testing.assert_almost_equal(
        (3.0 * q1).to_array(),
        np.array([3.0, 6.0, 9.0]),
    )

    np.testing.assert_almost_equal(
        (q1 / 2.0).to_array(),
        np.array([0.5, 1.0, 1.5]),
    )

    np.testing.assert_almost_equal(
        (q1 * q2).to_array(),
        np.array([2.0, 7.0, 10.0]),
    )

    np.testing.assert_almost_equal(
        (q1 / q2).to_array(),
        np.array([0.5, 0.25, 0.5]),
    )

    q1 = anacal.math.tnumber(1.0, 2.0, 3.0)
    e = np.exp(1.0)
    e2 = np.exp(2.0)
    np.testing.assert_almost_equal(
        anacal.math.exp(q1).to_array(),
        np.array([1.0, 2.0, 3.0]) * e,
    )
    np.testing.assert_almost_equal(
        anacal.math.exp(q2).to_array(),
        np.array([1.0, 3.0, 4.0]) * e2,
    )
    q1 = anacal.math.tnumber(2.0, 2.0, 3.0)
    np.testing.assert_almost_equal(
        anacal.math.pow(q1, 3).to_array(),
        np.array([8.0, 24.0, 36.0]),
    )
    return
