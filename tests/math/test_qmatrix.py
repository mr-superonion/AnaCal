import anacal
import numpy as np


def test_tmatrix6():
    tmatrix_zeros = anacal.math.tmatrix6()
    for qn in tmatrix_zeros.data:
        assert qn.v == 0


    matrix = np.random.rand(36).reshape((6, 6)) + np.eye(6)
    tmatrix = anacal.math.tmatrix6(matrix)
    matrix_flatten = matrix.flatten()
    for ii, qn in enumerate(tmatrix.data):
        assert qn.v == matrix_flatten[ii]

    matrix2 = np.random.rand(36).reshape((6, 6)) + np.eye(6) * 2.0

    tmatrix2 = anacal.math.tmatrix6(matrix2)

    res = tmatrix.transpose()
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix.T,
    )

    res = tmatrix * tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix @ matrix2,
    )

    res = tmatrix2 * tmatrix
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix2 @ matrix,
    )

    res = tmatrix2 * tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix2 @ matrix2,
    )

    res = tmatrix + tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix + matrix2,
    )

    res = tmatrix - tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix - matrix2,
    )

    qn = anacal.math.tnumber(2.6, 0, 0)
    res = qn - tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        2.6 - matrix2,
    )

    res = qn * tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        2.6 * matrix2,
    )

    res = qn + tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        2.6 + matrix2,
    )

    res = qn + tmatrix2 / qn / qn
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


def test_tmatrix8():
    tmatrix_zeros = anacal.math.tmatrix8()
    for qn in tmatrix_zeros.data:
        assert qn.v == 0


    matrix = np.random.rand(64).reshape((8, 8)) + np.eye(8)
    tmatrix = anacal.math.tmatrix8(matrix)
    matrix_flatten = matrix.flatten()
    for ii, qn in enumerate(tmatrix.data):
        assert qn.v == matrix_flatten[ii]

    res = tmatrix.transpose()
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix.T,
    )

    matrix2 = np.random.rand(64).reshape((8, 8)) + np.eye(8) * 2.0
    tmatrix2 = anacal.math.tmatrix8(matrix2)

    res = tmatrix * tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix @ matrix2,
    )

    res = tmatrix2 * tmatrix
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix2 @ matrix,
    )

    res = tmatrix2 * tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix2 @ matrix2,
    )

    res = tmatrix + tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix + matrix2,
    )

    res = tmatrix - tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        matrix - matrix2,
    )

    dd = 5.3
    qn = anacal.math.tnumber(dd, 0, 0)
    res = qn - tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        dd - matrix2,
    )

    res = qn * tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        dd * matrix2,
    )

    res = qn + tmatrix2
    np.testing.assert_almost_equal(
        res.to_array()[:, :, 0],
        dd + matrix2,
    )

    res = qn + tmatrix2 / qn / qn
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
