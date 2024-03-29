import gc
import time

import anacal
import fpfs
import galsim
import numpy as np

from . import mem_used, print_mem


def test_fpfs_measure():
    scale = 0.2
    ngrid = 512
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )
    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=32, ny=32, scale=scale)
        .array
    )

    nord = 4
    det_nrot = 4
    sigma_as = 0.53
    bound = 16

    task = fpfs.image.measure_source(
        psf_data,
        pix_scale=scale,
        sigma_arcsec=sigma_as,
        nord=nord,
        det_nrot=det_nrot,
    )

    gal_data = np.random.randn(ngrid, ngrid)

    pthres = 0.2
    pratio = 0.05
    std = 0.4

    cfpfs = anacal.fpfs.FpfsImage(
        nx=ngrid,
        ny=ngrid,
        scale=scale,
        sigma_arcsec=sigma_as,
        klim=task.klim / scale,
        psf_array=psf_data,
    )
    noise_array = np.zeros((1, 1))
    smooth_data = cfpfs.smooth_image(gal_array=gal_data, noise_array=noise_array)
    out1 = cfpfs.find_peaks(
        smooth_data,
        fthres=1.0,
        pthres=pthres,
        pratio=pratio,
        bound=bound,
        std_m00=std * scale**2.0,
        std_v=std * scale**2.0,
    )

    cov_element = np.ones((task.ncol, task.ncol)) * std**2.0
    out2 = task.detect_source(
        gal_data,
        psf_data,
        cov_element,
        fthres=1.0,
        pthres=pthres,
        pratio=pratio,
        bound=bound,
        noise_array=None,
    )
    np.testing.assert_almost_equal(out2, np.array(out1))
    cfpfs_mtask = anacal.fpfs.FpfsImage(
        nx=32,
        ny=32,
        scale=scale,
        sigma_arcsec=sigma_as,
        klim=task.klim / scale,
        psf_array=psf_data,
    )
    src1 = cfpfs_mtask.measure_sources(gal_data, filter_image=task.bfunc, det=out1)
    src2 = task.measure(gal_data, out2)
    np.testing.assert_almost_equal(src1, src2, decimal=5)
    return


if __name__ == "__main__":
    test_fpfs_measure()
