import jax
import logging
import numpy as np
import jax.numpy as jnp
import numpy.lib.recfunctions as rfn

from . import dtype
from . import fpfs
from . import impt

logging.basicConfig(
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S --- ",
    level=logging.INFO,
)


def fpfs_im2cat(data, method):
    ny = data.image.shape[0]
    nx = data.image.shape[1]
    npady = (ny - data.psf.shape[0]) // 2
    npadx = (nx - data.psf.shape[1]) // 2
    psf_array3 = np.pad(data.psf, ((npady, npady), (npadx, npadx)), mode="constant")
    # FPFS Tasks
    # noise cov task
    noise_task = fpfs.image.measure_noise_cov(
        data.psf,
        sigma_arcsec=method.sigma_as,
        sigma_detect=method.sigma_det,
        nnord=method.nnord,
        pix_scale=data.scale,
    )
    cov_elem = np.array(noise_task.measure(data.noise_pow))
    meas_task = fpfs.image.measure_source(
        data.psf,
        sigma_arcsec=method.sigma_as,
        sigma_detect=method.sigma_det,
        nnord=method.nnord,
        pix_scale=data.scale,
    )
    std_modes = np.sqrt(np.diagonal(cov_elem))
    idm00 = fpfs.catalog.indexes["m00"]
    idv0 = fpfs.catalog.indexes["v0"]
    # Temp fix for 4th order estimator
    if method.nnord == 6:
        idv0 += 1
    thres = 9.5 * std_modes[idm00] * data.scale**2.0
    thres2 = -1.5 * std_modes[idv0] * data.scale**2.0
    cutmag = data.mag_zero * 0.99
    thres_min = 10 ** ((data.mag_zero - cutmag) / 2.5) * data.scale**2.0
    thres2_min = -thres / 10
    thres = max(thres, thres_min)
    thres2 = max(thres2, thres2_min)

    coords = fpfs.image.detect_sources(
        data.image,
        psf_array3,
        sigmaf=meas_task.sigmaf,
        sigmaf_det=meas_task.sigmaf_det,
        thres=thres,
        thres2=thres2,
        bound=method.rcut - 1,
    )
    out = meas_task.measure(data.image, coords)
    out = meas_task.get_results(out)
    sel = (out["fpfs_M00"] + out["fpfs_M20"]) > 0.0
    out = out[sel]
    coords = coords[sel]
    coords = np.rec.fromarrays(coords.T, dtype=[("fpfs_y", "i4"), ("fpfs_x", "i4")])
    return dtype.FPFSCatalog(
        catalog=out,
        position=coords,
        cov_mat=cov_elem,
        nnord=method.nnord,
        rcut=method.rcut,
    )


def fpfs_cat2shear(data, method):
    e1, enoise1, res1, rnoise1 = impt.fpfs.future.prepare_func_e(
        cov_mat=data.cov_mat,
        ratio=method.ratio,
        c0=method.c0,
        c2=method.c2,
        alpha=method.alpha,
        beta=method.beta,
        snr_min=method.snr_min,
        noise_rev=method.noise_rev,
        g_comp=1,
    )

    e2, enoise2, res2, rnoise2 = impt.fpfs.future.prepare_func_e(
        cov_mat=data.cov_mat,
        ratio=method.ratio,
        c0=method.c0,
        c2=method.c2,
        alpha=method.alpha,
        beta=method.beta,
        snr_min=method.snr_min,
        noise_rev=method.noise_rev,
        g_comp=2,
    )

    def func(ss):
        e1_est = e1._obs_func(ss) - enoise1._obs_func(ss)
        res1_est = res1._obs_func(ss) - rnoise1._obs_func(ss)
        e2_est = e2._obs_func(ss) - enoise2._obs_func(ss)
        res2_est = res2._obs_func(ss) - rnoise2._obs_func(ss)
        return jnp.array([e1_est, res1_est, e2_est, res2_est])

    e_res = jax.lax.map(func, rfn.structured_to_unstructured(data.catalog))
    return e_res


def process_image(data, method):
    if data.dtype != "image":
        raise ValueError("data type is not image")
    if method.method == "fpfs":
        return fpfs_im2cat(data, method)
    else:
        raise ValueError("Do not support method type: %s" % method.method)


def measure_shear(data, method):
    if data.dtype != "catalog":
        raise ValueError("data type is not image")
    if method.method == "fpfs":
        return fpfs_cat2shear(data, method)
    else:
        raise ValueError("Do not support method type: %s" % method.method)
