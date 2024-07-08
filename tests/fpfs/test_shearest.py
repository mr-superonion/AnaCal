import anacal
import galsim
import numpy as np
import pytest

ngrid = 64
mag_zero = 27


def simulate_gal_psf(scale, seed, rcut, gcomp="g1", nrot=4):
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )

    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )
    psf_data = psf_data[
        ngrid // 2 - rcut : ngrid // 2 + rcut,
        ngrid // 2 - rcut : ngrid // 2 + rcut,
    ]
    gname = "%s-0" % gcomp
    gal_data = anacal.simulation.make_isolated_sim(
        gal_type="mixed",
        sim_method="fft",
        psf_obj=psf_obj,
        gname=gname,
        seed=seed,
        ny=ngrid,
        nx=ngrid * nrot,
        scale=scale,
        do_shift=False,
        buff=0,
        nrot_per_gal=nrot,
    )[0]

    # force detection at center
    indx = np.arange(ngrid // 2, ngrid * nrot, ngrid)
    indy = np.arange(ngrid // 2, ngrid, ngrid)
    inds = np.meshgrid(indy, indx, indexing="ij")
    coords = np.vstack(inds).T
    coords = [(cc[0], cc[1], True, 0) for cc in coords]
    return gal_data, psf_data, coords


def do_test(scale, seed, rcut, gcomp):
    if gcomp == "g1":
        g1 = -0.02
        g2 = 0.0
    elif gcomp == "g2":
        g1 = 0.0
        g2 = -0.02
    else:
        raise ValueError("gcomp should be g1 or g2")
    sigma_arcsec = 0.53
    mag_zero = 30.0

    nrot = 12
    gal_data, psf_data, coords = simulate_gal_psf(
        scale, seed, rcut, gcomp, nrot
    )
    nord = 4
    # Since we do not run detection
    # no detection weight
    det_nrot = -1
    mtask = anacal.fpfs.FpfsMeasure(
        psf_array=psf_data,
        mag_zero=mag_zero,
        pixel_scale=scale,
        sigma_arcsec=sigma_arcsec,
        nord=nord,
        det_nrot=det_nrot,
    )

    # Run as an exposure
    mms = mtask.run(
        gal_array=gal_data,
        det=coords,
    )

    std = 0.1
    cov_matrix = np.ones((9, 9)) * std**2.0 * scale**4.0
    cov_matrix = anacal.fpfs.table.Covariance(
        array=cov_matrix,
        nord=nord,
        det_nrot=det_nrot,
        mag_zero=mag_zero,
        pixel_scale=scale,
        sigma_arcsec=sigma_arcsec,
    )
    ctask = anacal.fpfs.CatalogTask(
        nord=nord,
        det_nrot=det_nrot,
        cov_matrix=cov_matrix,
    )
    ctask.update_parameters(
        snr_min=0.0,
        r2_min=-0.1,
        c0=4,
    )
    ells = ctask.run(catalog2=mms)
    g1_est = np.average(ells["e1_2"]) / np.average(ells["e1_g1_2"])
    g2_est = np.average(ells["e2_2"]) / np.average(ells["e2_g2_2"])
    assert np.abs(g1_est - g1) < 3e-5
    assert np.abs(g2_est - g2) < 3e-5

    # run as a list
    gal_list = [gal_data[:, i * ngrid : (i + 1) * ngrid] for i in range(nrot)]
    psf_list = [psf_data] * nrot
    mms = mtask.run(
        gal_array=gal_data,
        det=coords,
    )
    array2 = np.vstack(
        [
            mtask.run(gal_array=gal_list[i], psf=psf_list[i]).array
            for i in range(nrot)
        ]
    )
    np.testing.assert_almost_equal(mms.array, array2)
    return


@pytest.mark.parametrize("seed", [12, 23, 42])
def test_shear_estimation(seed):
    do_test(0.2, seed, 32, "g1")
    return
