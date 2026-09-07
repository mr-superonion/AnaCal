import anacal
import numpy as np
import numpy.lib.recfunctions as rfn

from ..fixtures import load

ngrid = 64
mag_zero = 27

# (pixel scale, x shift, y shift) of the pre-rendered cases in
# tests/data/fpfs_force_measure.fits: a sheared Moffat PSF and a sheared
# Gaussian galaxy offset by the shift.  Same order as
# make_fixtures.FORCE_CASES.
CASES = [
    (0.2, 0.0, 0.0),
    (0.2, 2.31, 0.43),
    (0.2, -2.35, 1.63),
    (0.164, -0.5, 1.5),
]
FIX = load("fpfs_force_measure")


def simulate_gal_psf(scale, shift_x, shift_y):
    i = CASES.index((scale, shift_x, shift_y))
    psf_array = FIX[f"psf_{i}"]
    gal_array = FIX[f"gal_{i}"]

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
    sigma_shapelets = 0.53

    gal_array, psf_array, coords = simulate_gal_psf(
        scale,
        shift_x,
        shift_y,
    )

    ftask = anacal.fpfs.FpfsTask(
        npix=64,
        pixel_scale=scale,
        sigma_shapelets=sigma_shapelets,
        psf_array=psf_array,
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
