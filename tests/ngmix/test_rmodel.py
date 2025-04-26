import anacal
import numpy as np


def test_ngmix_gaussian():
    flux = 1.4
    t = np.pi / 5.0
    a1 = 0.15
    a2 = 0.22
    x1 = 0.15
    x2 = -0.41
    sigma = 0.45
    scale = 1.0

    gauss_model = anacal.ngmix.NgmixGaussian()
    gauss_model.F = anacal.math.qnumber(flux)
    gauss_model.t = anacal.math.qnumber(t)
    gauss_model.a1 = anacal.math.qnumber(a1)
    gauss_model.a2 = anacal.math.qnumber(a2)
    gauss_model.x1 = anacal.math.qnumber(x1)
    gauss_model.x2 = anacal.math.qnumber(x2)
    kernel = gauss_model.prepare_modelD(scale=scale, sigma_arcsec=sigma)

    x = -0.43
    y = 0.21
    a = gauss_model.get_r2(x, y, kernel)

    res = np.array(
        [a.v.v, a.v_F.v, a.v_t.v, a.v_a1.v, a.v_a2.v, a.v_x1.v, a.v_x2.v]
    )
    res_target = np.array([
        2.87789694, 0.0, -0.0810202, -0.06508841, -4.96132116,  4.70115562,
        -4.88568326
    ])
    np.testing.assert_almost_equal(res, res_target)

    res = np.array(
        [kernel.f.v, kernel.f_a1.v, kernel.f_a2.v]
    )
    res_target = np.array(
        [0.6698515118823184, -0.4465676745882123, -0.5873548529856918]
    )
    np.testing.assert_almost_equal(res, res_target)

    a = gauss_model.get_model(x, y, kernel)
    res = np.array(
        [a.v.v, a.v_F.v, a.v_t.v, a.v_a1.v, a.v_a2.v, a.v_x1.v, a.v_x2.v]
    )
    res_target = np.array([
        0.22242275, 0.15887339, 0.00901037, -0.14104326, 0.35672543,
        -0.52282197, 0.54334354,
    ])
    np.testing.assert_almost_equal(res, res_target)
    return


# def get_r2(data):
#     flux, t, a1, a2, x1, x2 = data
#     # Rotation matrix
#     rot = jnp.array([
#         [jnp.cos(t), -jnp.sin(t)],
#         [jnp.sin(t),  jnp.cos(t)]
#     ])
#     # Diagonal eigenvalue matrix (variances)
#     eig = jnp.array([
#         [a1**2 + sigma**2, 0],
#         [0,    a2**2 + sigma**2]]
#     )
#     # Covariance matrix
#     mat = rot @ eig @ rot.T
#     mat_inv = jnp.linalg.inv(mat)
#     d = jnp.array([x - x1, y - x2])
#     return d.T @ mat_inv @ d

# def get_f(data):
#     flux, t, a1, a2, x1, x2 = data
#     # Rotation matrix
#     rot = jnp.array([
#         [jnp.cos(t), -jnp.sin(t)],
#         [jnp.sin(t),  jnp.cos(t)]
#     ])
#     # Diagonal eigenvalue matrix (variances)
#     eig = jnp.array([
#         [a1**2 + sigma**2, 0],
#         [0,    a2**2 + sigma**2]]
#     )
#     # Covariance matrix
#     mat = rot @ eig @ rot.T
#     return 1.0 / jnp.sqrt(jnp.linalg.det(mat)) / 2.0 / jnp.pi


# def get_model(data):
#     f = get_f(data)
#     flux, t, e1, e2, x1, x2 = data
#     r2 = get_r2(data)
#     return flux * f * jnp.exp(-0.5 * r2)

# data = jnp.array([flux, t, a1, a2, x1, x2])
# print(np.array([get_r2(data)] + list(jax.grad(get_r2)(data))))
# print(np.array([get_f(data)] + list(jax.grad(get_f)(data)[2:4])))
# print(jnp.diag(jax.hessian(get_f)(data))[2:4])
# print(np.array([get_model(data)] + list(jax.grad(get_model)(data))))
