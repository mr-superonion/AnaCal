import anacal
import fpfs
import galsim
import numpy as np
import pytest


@pytest.mark.parametrize("seed", [1, 43, 162])
def test_fpfs_measure(seed):
    scale = 0.2
    ngrid = 1024
    ngrid2 = 64
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
    mag_zero = 30.0
    sigma_as = 0.53
    bound = 32
    rng = np.random.RandomState(seed)
    gal_data = rng.randn(ngrid, ngrid)

    std = 0.4

    task = fpfs.image.measure_source(
        psf_data,
        pix_scale=scale,
        sigma_arcsec=sigma_as,
        nord=nord,
        det_nrot=det_nrot,
    )
    cov_element = np.ones((task.ncol, task.ncol)) * std**2.0
    cov_matrix_obj = anacal.fpfs.table.FpfsCovariance(
        array=cov_element,
        nord=nord,
        det_nrot=det_nrot,
        mag_zero=mag_zero,
        pixel_scale=scale,
        sigma_arcsec=sigma_as,
    )

    dtask = anacal.fpfs.FpfsDetect(
        nx=ngrid,
        ny=ngrid,
        psf_array=psf_data,
        mag_zero=mag_zero,
        pixel_scale=scale,
        sigma_arcsec=sigma_as,
        cov_matrix=cov_matrix_obj,
        det_nrot=det_nrot,
    )
    det1 = dtask.run(
        gal_array=gal_data,
        fthres=1.0,
        pthres=anacal.fpfs.fpfs_det_sigma2 + 0.02,
        bound=bound,
        noise_array=None,
    )
    mtask = anacal.fpfs.FpfsMeasure(
        psf_array=psf_data,
        mag_zero=mag_zero,
        pixel_scale=scale,
        sigma_arcsec=sigma_as,
        nord=nord,
        det_nrot=det_nrot,
    )
    src1 = mtask.run(gal_array=gal_data, det=det1)

    np.testing.assert_almost_equal(mtask.bfunc.real, task.bfunc.real)
    np.testing.assert_almost_equal(mtask.bfunc.imag, task.bfunc.imag)
    det2 = task.detect_source(
        gal_data,
        psf_data,
        cov_element,
        fthres=1.0,
        pthres=0.8,
        pratio=0.0,
        bound=bound,
        noise_array=None,
    )
    src2 = task.measure(gal_data, det2)
    np.testing.assert_almost_equal(src1.array, src2, decimal=5)

    psf_data2 = np.zeros((1, 1, ngrid2, ngrid2))
    psf_data2[0, 0] = psf_data
    grid_psf = anacal.psf.GridPsf(
        x0=0,
        y0=0,
        dx=ngrid,
        dy=ngrid,
        model_array=psf_data2,
    )
    src3 = mtask.run(gal_array=gal_data, psf=grid_psf, det=det1)
    np.testing.assert_almost_equal(src1.array, src3.array, decimal=5)
    return


if __name__ == "__main__":
    test_fpfs_measure(10)
