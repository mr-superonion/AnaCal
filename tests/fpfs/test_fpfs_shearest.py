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

    psf_array = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )
    psf_array = psf_array[
        ngrid // 2 - rcut : ngrid // 2 + rcut,
        ngrid // 2 - rcut : ngrid // 2 + rcut,
    ]
    gname = "%s-0" % gcomp
    gal_array = anacal.simulation.make_isolated_sim(
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
    coords = [(cc[0], cc[1]) for cc in coords]
    return gal_array, psf_array, coords


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

    nrot = 12
    gal_array, psf_array, coords = simulate_gal_psf(
        scale, seed, rcut, gcomp, nrot
    )

    ftask = anacal.fpfs.FpfsTask(
        npix=64,
        pixel_scale=scale,
        sigma_arcsec=sigma_arcsec,
        psf_array=psf_array,
        do_detection=False,
    )

    src = ftask.run(
        gal_array=gal_array,
        psf=psf_array,
        det=coords,
    )

    ells = anacal.fpfs.measure_fpfs(
        C0=4,
        x_array=src["data"],
        y_array=src["noise"],
    )

    # The 2nd order shear estimator
    g1_est = np.average(ells["e1"]) / np.average(ells["de1_dg1"])
    g2_est = np.average(ells["e2"]) / np.average(ells["de2_dg2"])
    assert np.abs(g1_est - g1) < 3e-5
    assert np.abs(g2_est - g2) < 3e-5

    # The 4th order shear estimator
    g1_est = np.average(ells["q1"]) / np.average(ells["dq1_dg1"])
    g2_est = np.average(ells["q2"]) / np.average(ells["dq2_dg2"])
    assert np.abs(g1_est - g1) < 3e-5
    assert np.abs(g2_est - g2) < 3e-5

    # run as a list
    gal_list = [gal_array[:, i * ngrid : (i + 1) * ngrid] for i in range(nrot)]
    psf_list = [psf_array] * nrot
    src_g = np.array(
        [
            ftask.run(gal_array=gal_list[i], psf=psf_list[i])["data"][0]
            for i in range(nrot)
        ],
        dtype=src["data"].dtype,
    )
    assert np.all(src["data"] == src_g)
    return


@pytest.mark.parametrize("seed", [12, 23, 42])
def test_shear_estimation(seed):
    do_test(0.2, seed, 32, "g1")
    return
