import anacal
import numpy as np


def test_ngmix_gaussian():
    A = 1.4
    t = 1.2
    e1 = 0.5
    e2 = -0.23
    x1 = 0.15
    x2 = -0.41
    sigma = 0.45
    scale = 1.0
    gauss_model = anacal.ngmix.NgmixGaussian()
    gauss_model.A = anacal.math.tnumber(A)
    gauss_model.t = anacal.math.tnumber(t)
    gauss_model.e1 = anacal.math.tnumber(e1)
    gauss_model.e2 = anacal.math.tnumber(e2)
    gauss_model.x1 = anacal.math.tnumber(x1)
    gauss_model.x2 = anacal.math.tnumber(x2)

    kernel = gauss_model.prepare_model(scale=scale, sigma_arcsec=sigma)
    np.testing.assert_almost_equal(
        np.array([kernel.f_t.v, kernel.f_e1.v, kernel.f_e2.v]),
        np.array([-0.03284224, 0.01147575, -0.00527885]),
    )

    np.testing.assert_almost_equal(
        np.array([kernel.f_tt.v, kernel.f_e1e1.v, kernel.f_e2e2.v]),
        np.array([0.0623498, 0.04639762, 0.0279127]),
    )

    x = -0.43
    y = 0.21
    a = gauss_model.get_r2(x, y, kernel)

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


    a = gauss_model.get_model(x, y, kernel)
    res = np.array(
        [a.v.v, a.v_A.v, a.v_t.v, a.v_e1.v, a.v_e2.v, a.v_x1.v, a.v_x2.v]
    )
    res_target = np.array(
        [0.022742572507201804, 0.01624469, -0.04270841, 0.01428684, -0.0076135,
         -0.00044416, 0.00227058]
    )
    np.testing.assert_almost_equal(res, res_target)

    res = np.array(
        [a.v_tt.v, a.v_e1e1.v, a.v_e2e2.v, a.v_x1x1.v, a.v_x2x2.v]
    )
    res_target = np.array(
        [0.07490922, 0.05550646, 0.03633384, -0.00144803, -0.00404018]
    )
    np.testing.assert_almost_equal(res, res_target)
    return


# import jax
# import jax.numpy as jnp

# def get_f(data):
#     A, t, e1, e2, x1, x2 = data
#     rho = jnp.exp(t)
#     mat = jnp.array(
#         [[rho**2 * (1+e1)+ sigma**2, rho**2 * e2],
#          [rho**2 * e2, rho**2 * (1-e1) + sigma**2]]
#     )
#     return 1.0 / jnp.sqrt(jnp.linalg.det(mat)) / 2.0 / jnp.pi

# def get_r2(data):
#     A, t, e1, e2, x1, x2 = data
#     rho = jnp.exp(t)
#     mat = jnp.array(
#       [[rho**2 * (1+e1)+ sigma**2, rho**2 * e2],
#          [rho**2 * e2, rho**2 * (1-e1) + sigma**2]]
#     )
#     mat_inv = jnp.linalg.inv(mat)
#     d = jnp.array([x - x1, y - x2])
#     return d.T @ mat_inv @ d

# def get_model(data):
#     f = get_f(data)
#     A, t, e1, e2, x1, x2 = data
#     rho = jnp.exp(t)
#     r2 = get_r2(data)
#     return A * f * jnp.exp(-0.5 * r2)

# data = jnp.array([A, t, e1, e2, x1, x2])

# print(get_f(data))
# print(kernel.f.v)
# print(kernel.f_t.v, kernel.f_e1.v, kernel.f_e2.v)
# print(kernel.f_tt.v, kernel.f_e1e1.v, kernel.f_e2e2.v)
# print("-----")
# print(jax.grad(get_f)(data))
# print(jnp.diag(jax.hessian(get_f)(data)))

# print(np.array([get_r2(data)] + list(jax.grad(get_r2)(data))))
# print(jnp.diag(jax.hessian(get_r2)(data))[1:])

# print(np.array([get_model(data)] + list(jax.grad(get_model)(data))))
# print(jnp.diag(jax.hessian(get_model)(data))[1:])
