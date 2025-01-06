import anacal
import numpy as np


def test_ngmix_gaussian():
    xc, yc = 0.82, -0.24
    GaussModel = anacal.ngmix.NgmixGaussian(sigma=0.52)
    A = anacal.math.qnumber(1.0, 0.0, 0.0, 0.0, 0.0)
    rho = anacal.math.qnumber(1.2, 0.0, 0.0, 0.0, 0.0)
    g1 = anacal.math.qnumber(0.05, 0.0, 0.0, 0.0, 0.0)
    g2 = anacal.math.qnumber(-0.08, 0.0, 0.0, 0.0, 0.0)
    x = anacal.math.qnumber(0.9, 0.0, 0.0, 0.0, 0.0)
    y = anacal.math.qnumber(-0.2, 0.0, 0.0, 0.0, 0.0)
    GaussModel.set_params([A, rho, g1, g2, x, y])

    image_val = anacal.math.qnumber(1.2, 0.0, 0.0, 0.0, 0.0)
    variance_val = 3.2
    loss = GaussModel.loss(image_val, variance_val, xc, yc)
    np.testing.assert_almost_equal(loss.v.v, 0.006956528938787841)

    res = np.array(
        [
            loss.v_A.v,
            loss.v_rho.v,
            loss.v_g1.v,
            loss.v_g2.v,
            loss.v_x.v,
            loss.v_y.v,
        ]
    )
    res_target = np.array(
        [
            -0.06521264,
            -0.00120239,
            -0.00073691,
            -0.00117906,
            0.01324967,
            0.00957248,
        ]
    )
    np.testing.assert_almost_equal(res, res_target)

    res = np.array(
        [
            loss.v_AA.v,
            loss.v_rhorho.v,
            loss.v_g1g1.v,
            loss.v_g2g2.v,
            loss.v_xx.v,
            loss.v_yy.v,
        ]
    )
    res_target = np.array(
        [0.30566166, 0.00308773, 0.00137054, 0.00141844, 0.16214834, 0.19089936]
    )
    np.testing.assert_almost_equal(res, res_target)

    m = GaussModel.model(xc, yc)
    np.testing.assert_almost_equal(m.v.v, 0.9889981393251657)

    res = np.array([m.v_A.v, m.v_rho.v, m.v_g1.v, m.v_g2.v, m.v_x.v, m.v_y.v])
    res_target = np.array(
        [
            0.9889981393251657,
            0.0182351950512148,
            0.011175809020152094,
            0.017881294432243306,
            -0.20094104618233422,
            -0.14517375917177527,
        ]
    )
    np.testing.assert_almost_equal(res, res_target)

    res = np.array(
        [m.v_AA.v, m.v_rhorho.v, m.v_g1g1.v, m.v_g2g2.v, m.v_xx.v, m.v_yy.v]
    )
    res_target = np.array(
        [
            0.0,
            -0.04525176622848741,
            -0.020193364652450762,
            -0.019996355198569876,
            -2.2677400784513333,
            -2.795248000180942,
        ]
    )
    np.testing.assert_almost_equal(res, res_target)

    r2 = GaussModel.get_r2(xc, yc)
    np.testing.assert_almost_equal(r2.v.v, 0.005982777777777785)

    res = np.array([r2.v_rho.v, r2.v_g1.v, r2.v_g2.v, r2.v_x.v, r2.v_y.v])
    res_target = np.array(
        [
            -0.009971296296296308,
            -0.00611111111111113,
            -0.009777777777777783,
            0.10987777777777787,
            0.07938333333333333,
        ]
    )
    np.testing.assert_almost_equal(res, res_target)

    res = np.array(
        [r2.v_rhorho.v, r2.v_g1g1.v, r2.v_g2g2.v, r2.v_xx.v, r2.v_yy.v]
    )
    res_target = np.array(
        [
            0.02492824074074077,
            0.011111111111111127,
            0.011111111111111127,
            1.2623611111111113,
            1.5401388888888892,
        ]
    )
    np.testing.assert_almost_equal(res, res_target)
    return
