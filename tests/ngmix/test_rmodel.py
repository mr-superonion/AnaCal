import anacal
import numpy as np


def test_ngmix_gaussian():
    A = 1.4
    rho = 2.4
    e1 = 0.5
    e2 = -0.23
    x0 = 0.15
    y0 = -0.41
    sigma = 0.45
    params0 = [
        anacal.math.qnumber(A, 0.0, 0.0, 0.0, 0.0),
        anacal.math.qnumber(rho, 0.0, 0.0, 0.0, 0.0),
        anacal.math.qnumber(e1, 0.0, 0.0, 0.0, 0.0),
        anacal.math.qnumber(e2, 0.0, 0.0, 0.0, 0.0),
        anacal.math.qnumber(x0, 0.0, 0.0, 0.0, 0.0),
        anacal.math.qnumber(y0, 0.0, 0.0, 0.0, 0.0),
    ]
    gauss_model = anacal.ngmix.NgmixGaussian(sigma_arcsec=sigma)
    gauss_model.set_params(params0)
    gauss_model.prepare_grad()
    x = -0.43
    y = 0.21
    a = gauss_model.get_r2(x, y)

    res = np.array(
        [a.v.v, a.v_A.v, a.v_rho.v, a.v_e1.v, a.v_e2.v, a.v_x.v, a.v_y.v]
    )
    res_target = np.array(
        [0.13658638410732674, 0.0, -0.10781366231595958, 0.18853835611637842,
         0.08070187803052695, 0.07579620147222685, -0.369695115098003]
    )
    np.testing.assert_almost_equal(res, res_target)

    res = np.array(
        [a.v_rhorho.v, a.v_e1e1.v, a.v_e2e2.v, a.v_xx.v, a.v_yy.v]
    )
    res_target = np.array(
        [0.1253963951060485, 0.8459647194611971, 0.25880106483425097,
         0.24174658711785238, 0.6934774360388029]
    )
    np.testing.assert_almost_equal(res, res_target)


    a = gauss_model.model(x, y)
    res = np.array(
        [a.v.v, a.v_A.v, a.v_rho.v, a.v_e1.v, a.v_e2.v, a.v_x.v, a.v_y.v]
    )
    res_target = np.array(
        [1.3075812343742934, 0.9339865959816382, 0.07048756082675783,
         -0.1232646082087771, -0.05276213064574005, -0.04955484534096851,
         0.24170319747099664]
    )
    res = np.array(
        [a.v_rhorho.v, a.v_e1e1.v, a.v_e2e2.v, a.v_xx.v, a.v_yy.v]
    )


#     def get_r2(data):
#         A, rho, e1, e2, x0, y0 = data
#         mat = jnp.array(
#             [[rho**2 * (1+e1)+ sigma**2, rho**2 * e2],
#              [rho**2 * e2, rho**2 * (1-e1) + sigma**2]]
#         )
#         mat_inv = jnp.linalg.inv(mat)
#         d = jnp.array([x - x0, y - y0])
#         return d.T @ mat_inv @ d

#     def get_model(data):
#         A, rho, e1, e2, x0, y0 = data
#         r2 = get_r2(data)
#         return A * jnp.exp(-0.5 * r2)

#     data = jnp.array([A, rho, e1, e2, x0, y0])

#     print(get_r2(data))
#     print(jax.grad(get_r2)(data))
#     print(jnp.diag(jax.hessian(get_r2)(data)))

#     print(get_model(data))
#     print(jax.grad(get_model)(data))
#     print(jnp.diag(jax.hessian(get_model)(data)))

    return
