import anacal
import numpy as np


def test_ngmix_gaussian():
    A = 1.4
    z = 1.2
    e1 = 0.5
    e2 = -0.23
    x1 = 0.15
    x2 = -0.41
    sigma = 0.45
    params0 = anacal.ngmix.modelNumber(
        anacal.math.qnumber(A, 0.0, 0.0, 0.0, 0.0),
        anacal.math.qnumber(z, 0.0, 0.0, 0.0, 0.0),
        anacal.math.qnumber(e1, 0.0, 0.0, 0.0, 0.0),
        anacal.math.qnumber(e2, 0.0, 0.0, 0.0, 0.0),
        anacal.math.qnumber(x1, 0.0, 0.0, 0.0, 0.0),
        anacal.math.qnumber(x2, 0.0, 0.0, 0.0, 0.0),
    )
    gauss_model = anacal.ngmix.NgmixGaussian(sigma_arcsec=sigma)
    gauss_model.set_params(params0)
    gauss_model.prepare_grad()
    x = -0.43
    y = 0.21
    a = gauss_model.get_r2(x, y)

    res = np.array(
        [a.v.v, a.v_A.v, a.v_t.v, a.v_e1.v, a.v_e2.v, a.v_x1.v, a.v_x2.v]
    )
    res_target = np.array(
        [0.07322712, 0.0, -0.14226285, 0.10567132, 0.04298667, 0.03905975,
         -0.19967676]
    )
    np.testing.assert_almost_equal(res, res_target)

    res = np.array(
        [a.v_tt.v, a.v_e1e1.v, a.v_e2e2.v, a.v_x1x1.v, a.v_x2x2.v]
    )
    res_target = np.array(
        [0.26834528, 0.48734336, 0.14561384, 0.12810377, 0.37523164]
    )
    np.testing.assert_almost_equal(res, res_target)


    a = gauss_model.model(x, y)
    res = np.array(
        [a.v.v, a.v_A.v, a.v_t.v, a.v_e1.v, a.v_e2.v, a.v_x1.v, a.v_x2.v]
    )
    res_target = np.array(
        [1.34966805, 0.96404861, 0.09600381, -0.0713106, -0.02900887,
         -0.02635885, 0.13474867]
    )
    np.testing.assert_almost_equal(res, res_target)

    res = np.array(
        [a.v_tt.v, a.v_e1e1.v, a.v_e2e2.v, a.v_x1x1.v, a.v_x2x2.v]
    )
    res_target = np.array(
        [-0.17425964, -0.32510814, -0.09764168, -0.085934, -0.23976599]
    )
    np.testing.assert_almost_equal(res, res_target)
    return


# import jax
# import jax.numpy as jnp

# def get_r2(data):
#     A, z, e1, e2, x1, x2 = data
#     rho = jnp.exp(z)
#     mat = jnp.array(
#         [[rho**2 * (1+e1)+ sigma**2, rho**2 * e2],
#          [rho**2 * e2, rho**2 * (1-e1) + sigma**2]]
#     )
#     mat_inv = jnp.linalg.inv(mat)
#     d = jnp.array([x - x1, y - x2])
#     return d.T @ mat_inv @ d

# def get_model(data):
#     A, z, e1, e2, x1, x2 = data
#     rho = jnp.exp(z)
#     r2 = get_r2(data)
#     return A * jnp.exp(-0.5 * r2)

# data = jnp.array([A, z, e1, e2, x1, x2])

# print(np.array([get_r2(data)] + list(jax.grad(get_r2)(data))))
# print(jnp.diag(jax.hessian(get_r2)(data))[1:])

# print(np.array([get_model(data)] + list(jax.grad(get_model)(data))))
# print(jnp.diag(jax.hessian(get_model)(data))[1:])
