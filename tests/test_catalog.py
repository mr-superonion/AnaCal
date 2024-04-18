import numpy as np
from anacal.fpfs import FpfsCatalog
from fpfs.catalog import fpfs_catalog


def test_catalog():
    mm = np.abs(np.random.randn(1000, 21) * 10)
    nn = np.random.randn(1000, 21) * 4
    cov_mat = 2.0 + np.random.randn(21, 21)

    det_nrot = 4
    snr_min = 12
    r2_min = 0.1
    c0 = 5.0
    c2 = 22.74
    alpha = 1.0
    beta = 0.0
    pthres = 0.8
    pratio = 0.00

    cat_obj = fpfs_catalog(
        cov_mat=cov_mat,
        snr_min=snr_min,
        r2_min=r2_min,
        ratio=1.6,
        c0=c0,
        c2=c2,
        alpha=alpha,
        beta=beta,
        pthres=pthres,
        pratio=pratio,
        det_nrot=det_nrot,
    )

    cat_obj2 = FpfsCatalog(
        cov_mat=cov_mat,
        snr_min=snr_min,
        r2_min=r2_min,
        c0=c0,
        c2=c2,
        alpha=alpha,
        beta=beta,
        pthres=pthres,
        pratio=pratio,
        det_nrot=det_nrot,
    )

    out1 = cat_obj.measure_g1_renoise(mm, nn)
    out2 = cat_obj2.measure_g1_renoise(mm, nn)
    np.testing.assert_array_almost_equal(out1, out2)
    return
