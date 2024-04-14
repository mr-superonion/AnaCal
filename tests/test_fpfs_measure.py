import anacal
import fpfs
import galsim
import numpy as np
import pytest


@pytest.mark.parametrize("seed", [1, 43, 162])
def test_fpfs_measure(seed):
    scale = 0.2
    ngrid = 1024
    ngrid2 = 32
    psf_obj = galsim.Moffat(
        beta=3.5,
        fwhm=0.6,
        trunc=0.6 * 4.0,
    ).shear(e1=0.02, e2=-0.02)
    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(
            nx=ngrid2,
            ny=ngrid2,
            scale=scale,
        )
        .array
    )

    nord = 4
    det_nrot = 4
    sigma_as = 0.53
    bound = 16
    rng = np.random.RandomState(seed)
    gal_data = rng.randn(ngrid, ngrid)

    pthres = 0.2
    pratio = 0.05
    std = 0.4

    dtask = anacal.fpfs.FpfsDetect(
        nx=ngrid,
        ny=ngrid,
        psf_array=psf_data,
        pix_scale=scale,
        sigma_arcsec=sigma_as,
        det_nrot=det_nrot,
    )
    out1 = dtask.run(
        gal_array=gal_data,
        fthres=1.0,
        pthres=pthres,
        pratio=pratio,
        bound=bound,
        std_m00=std,
        std_v=std,
        noise_array=None,
    )
    mtask = anacal.fpfs.FpfsMeasure(
        psf_array=psf_data,
        pix_scale=scale,
        sigma_arcsec=sigma_as,
        det_nrot=det_nrot,
    )
    src1 = mtask.run(gal_array=gal_data, det=out1)

    task = fpfs.image.measure_source(
        psf_data,
        pix_scale=scale,
        sigma_arcsec=sigma_as,
        nord=nord,
        det_nrot=det_nrot,
    )

    np.testing.assert_almost_equal(mtask.bfunc.real, task.bfunc.real)
    np.testing.assert_almost_equal(mtask.bfunc.imag, task.bfunc.imag)
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
    src2 = task.measure(gal_data, out2)
    np.testing.assert_almost_equal(src1, src2, decimal=5)
    return


if __name__ == "__main__":
    test_fpfs_measure()
