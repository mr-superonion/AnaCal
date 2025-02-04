import anacal
import numpy as np


def test_qmatrix6():
    qmatrix_zeros = anacal.math.qmatrix6()
    for qn in qmatrix_zeros.data:
        assert qn.v == 0


    matrix = np.random.rand(36).reshape((6, 6)) + np.eye(6)
    qmatrix = anacal.math.qmatrix6(matrix)
    matrix_flatten = matrix.flatten()
    for ii, qn in enumerate(qmatrix.data):
        assert qn.v == matrix_flatten[ii]

    matrix2 = np.random.rand(36).reshape((6, 6)) + np.eye(6) * 2.0

    qmatrix2 = anacal.math.qmatrix6(matrix2)

    res = qmatrix.transpose()
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix.T,
    )

    res = qmatrix * qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix @ matrix2,
    )

    res = qmatrix2 * qmatrix
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix2 @ matrix,
    )

    res = qmatrix2 * qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix2 @ matrix2,
    )

    res = qmatrix + qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix + matrix2,
    )

    res = qmatrix - qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix - matrix2,
    )

    qn = anacal.math.qnumber(2.6, 0, 0, 0, 0)
    res = qn - qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        2.6 - matrix2,
    )

    res = qn * qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        2.6 * matrix2,
    )

    res = qn + qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        2.6 + matrix2,
    )

    res = qn + qmatrix2 / qn / qn
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        2.6 + matrix2 / 2.6 / 2.6,
    )

    eye = anacal.math.eye6()
    np.testing.assert_almost_equal(
        eye.to_array()[:, :, 0],
        np.eye(6),
    )
    return


def test_qmatrix8():
    qmatrix_zeros = anacal.math.qmatrix8()
    for qn in qmatrix_zeros.data:
        assert qn.v == 0


    matrix = np.random.rand(64).reshape((8, 8)) + np.eye(8)
    qmatrix = anacal.math.qmatrix8(matrix)
    matrix_flatten = matrix.flatten()
    for ii, qn in enumerate(qmatrix.data):
        assert qn.v == matrix_flatten[ii]

    res = qmatrix.transpose()
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix.T,
    )

    matrix2 = np.random.rand(64).reshape((8, 8)) + np.eye(8) * 2.0
    qmatrix2 = anacal.math.qmatrix8(matrix2)

    res = qmatrix * qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix @ matrix2,
    )

    res = qmatrix2 * qmatrix
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix2 @ matrix,
    )

    res = qmatrix2 * qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix2 @ matrix2,
    )

    res = qmatrix + qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix + matrix2,
    )

    res = qmatrix - qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix - matrix2,
    )

    dd = 5.3
    qn = anacal.math.qnumber(dd, 0, 0, 0, 0)
    res = qn - qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        dd - matrix2,
    )

    res = qn * qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        dd * matrix2,
    )

    res = qn + qmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        dd + matrix2,
    )

    res = qn + qmatrix2 / qn / qn
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        dd + matrix2 / dd / dd,
    )

    eye = anacal.math.eye8()
    np.testing.assert_almost_equal(
        eye.to_array()[:, :, 0],
        np.eye(8),
    )
    return
