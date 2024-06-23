import anacal
import galsim
import numpy as np
from fpfs.catalog import fpfs_catalog


def test_catalog_shear():
    mm = np.abs(np.random.randn(1000, 21) * 10)
    nn = np.random.randn(1000, 21) * 4
    cov_matrix = np.abs(2.0 + np.random.randn(21, 21))

    nord = 4
    det_nrot = 4
    snr_min = 12
    r2_min = 0.1
    c0 = 5.0
    mag_zero = 30
    sigma_arcsec = 0.52
    pixel_scale = 0.2

    cat_obj = fpfs_catalog(
        cov_mat=cov_matrix,
        snr_min=snr_min,
        r2_min=r2_min,
        ratio=1.6,
        c0=c0,
        c2=100,
        alpha=1.0,
        beta=0.0,
        pthres=0.8,
        pratio=0.0,
        det_nrot=det_nrot,
    )

    nshapelets = len(cat_obj.name_shapelets)
    cov_s = anacal.fpfs.table.FpfsCovariance(
        array=cov_matrix[:nshapelets, :nshapelets],
        mag_zero=mag_zero,
        pixel_scale=pixel_scale,
        nord=nord,
        sigma_arcsec=sigma_arcsec,
    )
    cov_d = anacal.fpfs.table.FpfsCovariance(
        array=cov_matrix[nshapelets:, nshapelets:],
        mag_zero=mag_zero,
        pixel_scale=pixel_scale,
        det_nrot=det_nrot,
        sigma_arcsec=sigma_arcsec,
    )
    out1_1 = cat_obj.measure_g1_renoise(mm, nn)
    out1_2 = cat_obj.measure_g2_renoise(mm, nn)

    meas_obj = anacal.fpfs.ctask.CatalogTask(
        nord=nord,
        det_nrot=det_nrot,
        cov_matrix_s=cov_s,
        cov_matrix_d=cov_d,
    )

    meas_obj.update_parameters(
        snr_min=snr_min,
        r2_min=r2_min,
        c0=c0,
        pthres=0.12,
    )
    src_s = anacal.fpfs.table.FpfsCatalog(
        array=mm[:, :nshapelets],
        noise=nn[:, :nshapelets],
        mag_zero=mag_zero,
        pixel_scale=pixel_scale,
        nord=nord,
        sigma_arcsec=sigma_arcsec,
    )
    src_d = anacal.fpfs.table.FpfsCatalog(
        array=mm[:, nshapelets:],
        noise=nn[:, nshapelets:],
        mag_zero=mag_zero,
        pixel_scale=pixel_scale,
        det_nrot=det_nrot,
        sigma_arcsec=sigma_arcsec,
    )

    out2 = meas_obj.run(shapelet=src_s, detection=src_d)
    np.testing.assert_array_almost_equal(
        out2["wdet"] * out2["e1"],
        out1_1[:, 0],
    )
    np.testing.assert_array_almost_equal(
        out2["wdet_g1"] * out2["e1"] + out2["wdet"] * out2["e1_g1"],
        out1_1[:, 1],
    )
    np.testing.assert_array_almost_equal(
        out2["wdet"] * out2["e2"],
        out1_2[:, 0],
    )
    np.testing.assert_array_almost_equal(
        out2["wdet_g2"] * out2["e2"] + out2["wdet"] * out2["e2_g2"],
        out1_2[:, 1],
    )
    return


def test_catalog_mag():
    pixel_scale = 0.2
    psf_obj = galsim.Moffat(fwhm=0.8, beta=2.5)
    ngrid = 256
    psf_data = (
        psf_obj.shift(pixel_scale * 0.5, pixel_scale * 0.5)
        .drawImage(
            nx=64,
            ny=64,
            scale=pixel_scale,
            method="no_pixel",
        )
        .array
    )
    gal_obj = galsim.Convolve([galsim.Gaussian(sigma=0.52, flux=100), psf_obj])
    gal_data = (
        gal_obj.shift(pixel_scale * 0.5, pixel_scale * 0.5)
        .drawImage(
            nx=ngrid,
            ny=ngrid,
            scale=pixel_scale,
            method="no_pixel",
        )
        .array
    )

    nord = 4
    det_nrot = 4
    std = 1
    bound = 0
    sigma_arcsec = 0.52
    mag_zero = 30.0
    pthres = 0.2

    cov_matrix = np.ones((21, 21)) * std**2.0 * pixel_scale**4.0
    cov_matrix = anacal.fpfs.table.FpfsCovariance(
        array=cov_matrix,
        nord=nord,
        det_nrot=det_nrot,
        mag_zero=mag_zero,
        pixel_scale=pixel_scale,
        sigma_arcsec=sigma_arcsec,
    )
    dtask = anacal.fpfs.FpfsDetect(
        mag_zero=mag_zero,
        nx=gal_data.shape[1],
        ny=gal_data.shape[0],
        psf_array=psf_data,
        pixel_scale=pixel_scale,
        sigma_arcsec=sigma_arcsec,
        cov_matrix=cov_matrix,
        det_nrot=det_nrot,
    )
    det = dtask.run(
        gal_array=gal_data,
        fthres=1.0,
        pthres=pthres,
        bound=bound,
        noise_array=None,
    )

    mtask1 = anacal.fpfs.FpfsMeasure(
        mag_zero=mag_zero,
        psf_array=psf_data,
        pixel_scale=pixel_scale,
        sigma_arcsec=sigma_arcsec,
        nord=nord,
        det_nrot=-1,
    )
    src_s = mtask1.run(gal_array=gal_data, det=det)

    mtask2 = anacal.fpfs.FpfsMeasure(
        mag_zero=mag_zero,
        psf_array=psf_data,
        pixel_scale=pixel_scale,
        sigma_arcsec=sigma_arcsec,
        nord=-1,
        det_nrot=det_nrot,
    )
    src_d = mtask2.run(gal_array=gal_data, det=det)

    cat_obj = anacal.fpfs.CatalogTask(
        cov_matrix_s=cov_matrix,
        cov_matrix_d=cov_matrix,
        det_nrot=det_nrot,
        nord=nord,
    )

    cat_obj.update_parameters(
        snr_min=12,
        r2_min=0.1,
        c0=4,
        pthres=pthres,
    )
    flux1 = cat_obj.run(src_s, src_d)["flux"]
    flux2 = np.sum(gal_data)
    assert (flux1 - flux2) / flux2 < 0.01

    return
