import galsim
import anacal
import numpy as np
shear_value = 0.03
# Accurate to third order in shear
thres = shear_value ** 3.


def test_sims():
    scale = 0.168
    nx, ny = 64, 64
    psf_obj = galsim.Moffat(
        beta=3.5, fwhm=0.6, trunc=0.6 * 4.0
    ).shear(e1=0.02, e2=-0.02)

    seed = 212
    gname = "g1-2222"
    data_obj = anacal.simulation.make_isolate_sim(
        shear_value=shear_value,
        gal_type="basic",
        psf_obj=psf_obj,
        gname=gname,
        seed=seed,
        ny=ny * 1,
        nx=nx * 4,
        scale=scale,
        do_shift=False,
        nrot=4,
    )
    method_obj = anacal.dtype.FPFSMethod(
        sigma_as=0.5,
        sigma_det=0.5,
        rcut=32,
        nnord=4,
        noise_rev=False,
    )
    cat_obj = anacal.process.process_image(data_obj, method_obj)
    # test with FPFS
    ells = anacal.fpfs.catalog.fpfs_m2e(cat_obj.catalog, const=20)
    # test for g1
    shear1 = np.average(ells['fpfs_e1']) / np.average(ells['fpfs_R1E'])
    assert np.all(np.abs(shear1 - shear_value) < thres)
    # test for g2
    shear2 = np.average(ells['fpfs_e2']) / np.average(ells['fpfs_R2E'])
    assert np.all(np.abs(shear2 - 0.) < thres)

    # test the wrapped impt.FPFS function
    outcome = anacal.process.measure_shear(cat_obj, method_obj)
    shear1 = np.sum(outcome[:, 0]) / np.sum(outcome[:, 1])
    assert np.all(np.abs(shear1 - shear_value) < thres)
    return


if __name__ == "__main__":
    test_sims()
