import anacal
import galsim
import numpy as np
import numpy.lib.recfunctions as rfn

import fpfs


def test_fpfs_measure():
    seed = 12
    scale = 0.2
    ngrid = 1024
    ngrid2 = 64
    psf_obj = galsim.Moffat(
        beta=3.5,
        fwhm=0.6,
        trunc=0.6 * 4.0,
    ).shear(e1=0.02, e2=-0.02)
    psf_array = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(
            nx=ngrid2,
            ny=ngrid2,
            scale=scale,
        )
        .array
    )

    norder = 6
    det_nrot = 4
    sigma_arcsec = 0.53
    bound = 35
    rng = np.random.RandomState(seed)
    gal_data = rng.randn(ngrid, ngrid) * 5

    std = 0.2

    task = fpfs.image.measure_source(
        psf_array,
        pix_scale=scale,
        sigma_arcsec=sigma_arcsec,
        nord=norder,
        det_nrot=det_nrot,
    )

    kernel = anacal.fpfs.FpfsKernel(
        npix=64,
        pixel_scale=scale,
        sigma_arcsec=sigma_arcsec,
        psf_array=psf_array,
        compute_detect_kernel=True,
    )

    kernel.prepare_fpfs_bases()
    kernel.prepare_covariance(variance=std**2.0 * 2.0)
    cov_element = kernel.cov_matrix.array

    dtask = anacal.fpfs.FpfsDetect(
        kernel=kernel,
        bound=bound,
    )
    det1 = dtask.run(
        gal_array=gal_data,
        fthres=10.0,
        pthres=anacal.fpfs.fpfs_det_sigma2 + 0.02,
        noise_array=None,
    )

    det1 = det1[np.lexsort((det1["y"], det1["x"]))]
    mtask = anacal.fpfs.FpfsMeasure(
        kernel=kernel,
    )
    src_g, src_n = mtask.run(
        gal_array=gal_data,
        psf=psf_array,
        det=det1,
    )

    np.testing.assert_almost_equal(kernel.bfunc.real, task.bfunc.real)
    np.testing.assert_almost_equal(kernel.bfunc.imag, task.bfunc.imag)
    det2 = task.detect_source(
        gal_data,
        psf_array,
        cov_element,
        fthres=10.0,
        pthres=0.8,
        pratio=0.0,
        bound=bound,
        noise_array=None,
    )
    src1 = rfn.structured_to_unstructured(src_g)
    det2 = det2[np.lexsort((det2[:, 0], det2[:, 1]))]
    src2 = task.measure(gal_data, det2)
    np.testing.assert_almost_equal(
        src1,
        src2,
        decimal=4
    )

    # use grid PSF
    psf_array2 = np.zeros((1, 1, ngrid2, ngrid2))
    psf_array2[0, 0] = psf_array
    grid_psf = anacal.psf.GridPsf(
        x0=0,
        y0=0,
        dx=ngrid,
        dy=ngrid,
        model_array=psf_array2,
    )

    src_g3, src_n3 = mtask.run(gal_array=gal_data, psf=grid_psf, det=det1)
    src3 = rfn.structured_to_unstructured(src_g3)
    np.testing.assert_almost_equal(src1, src3, decimal=5)
    return


if __name__ == "__main__":
    test_fpfs_measure()
