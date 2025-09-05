import anacal
import galsim
import numpy as np
import numpy.lib.recfunctions as rfn

ngrid = 64
mag_zero = 27


def simulate_gal_psf(scale, shift_x, shift_y):
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.6, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )

    psf_array = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )

    gal_obj = galsim.Gaussian(fwhm=0.6).shear(e1=0.2, e2=-0.24)
    gal_array = (
        gal_obj.shift((0.5 + shift_x) * scale, (0.5 + shift_y) * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array
    )

    # force detection at center
    coords = np.array(
        [(ngrid / 2.0 + shift_y, ngrid / 2.0 + shift_x)],
        dtype=[
            ("y", "f8"),
            ("x", "f8"),
        ],
    )
    return gal_array, psf_array, coords


def do_test(scale, shift_x, shift_y):
    sigma_arcsec = 0.53

    gal_array, psf_array, coords = simulate_gal_psf(
        scale,
        shift_x,
        shift_y,
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

    return rfn.structured_to_unstructured(src["data"])


def test_shear_estimation():
    data1 = do_test(0.2, 0.0, 0.0)
    data2 = do_test(0.2, 2.31, 0.43)
    np.testing.assert_almost_equal(data1, data2, decimal=4)

    data2 = do_test(0.2, -2.35, 1.63)
    np.testing.assert_almost_equal(data1, data2, decimal=4)

    data2 = do_test(0.164, -0.5, 1.5)
    np.testing.assert_almost_equal(data1, data2, decimal=4)
    return
