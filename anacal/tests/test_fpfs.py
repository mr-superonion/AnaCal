import galsim
import anacal
import numpy as np


def test_noisy_gals():
    shear_value = 0.03
    scale = 0.168
    nx, ny = 64, 64
    psf_obj = galsim.Moffat(
        beta=3.5, fwhm=0.6, trunc=0.6 * 4.0
    ).shear(e1=0.02, e2=-0.02)

    seed = 212
    gname = "g1-2222"
    data_obj = anacal.simulation.make_isolate_sim(
        gal_type="basic",
        psf_obj=psf_obj,
        gname=gname,
        seed=seed,
        ny=ny,
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
    )
    print(data_obj.image.max())
    cat_obj = anacal.process.process_image(data_obj, method_obj)
    print(cat_obj.catalog)
    print(cat_obj.position)
    # assert np.all(np.abs(shear - shear_value) < thres)
    return


if __name__ == "__main__":
    test_noisy_gals()
