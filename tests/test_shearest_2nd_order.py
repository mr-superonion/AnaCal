import fpfs
import anacal
import galsim
import pytest
import numpy as np
import jax.numpy as jnp

ngrid = 64

def simulate_gal_psf(scale, seed, rcut, gcomp="g1", nrot=12):
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )

    psf_data = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )
    psf_data = psf_data[
        ngrid // 2 - rcut : ngrid // 2 + rcut, ngrid // 2 - rcut : ngrid // 2 + rcut
    ]
    gname = "%s-0" % gcomp
    gal_data = fpfs.simulation.make_isolate_sim(
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
    coords = [(cc[0], cc[1], True) for cc in coords]
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

    nrot = 12
    gal_data, psf_data, coords = simulate_gal_psf(scale, seed, rcut, gcomp, nrot)
    nord = 4
    det_nrot = 4
    mtask = anacal.fpfs.FpfsMeasure(
        psf_array=psf_data,
        pix_scale=scale,
        sigma_arcsec=0.53,
        nord=nord,
        det_nrot=det_nrot,
    )

    # Run as an exposure
    mms = mtask.run(
        gal_array=gal_data,
        det = coords,
    )
    mms = mtask.get_results(mms)
    ells = anacal.fpfs.catalog.m2e(mms, const=8)
    g1_est = np.average(ells["fpfs_e1"]) / np.average(ells["fpfs_R1E"])
    g2_est = np.average(ells["fpfs_e2"]) / np.average(ells["fpfs_R2E"])
    assert np.abs(g1_est - g1) < 3e-5
    assert np.abs(g2_est - g2) < 3e-5

    # run as a list
    gal_list = [gal_data[:, i* ngrid: (i+1)*ngrid] for i in range(nrot)]
    psf_list = [psf_data] * nrot
    mms = mtask.run(
        gal_array=gal_data,
        det = coords,
    )
    mms2 = np.vstack([
        mtask.run(
            gal_array=gal_list[i],
            psf_array=psf_list[i]
        ) for i in range(nrot)
    ])
    np.testing.assert_almost_equal(mms, mms2)
    return


@pytest.mark.parametrize("seed", [12, 23, 42])
def test_shear_estimation(seed):
    do_test(0.168, seed, 32, "g1")
    return

