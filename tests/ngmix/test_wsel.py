import anacal
import numpy as np

from ..fixtures import load

kwargs = {
    "sigma_arcsec": 0.4,
    "snr_min": 10,
    "variance": 1.75e-2,
    "omega_f": 0.1,
    "omega_v": 0.011,
}


# The magnitudes below sit inside the wsel transition, where dwsel/dg is
# non-zero.  That transition moved brighter (was ~27.5-27.9) when the second
# detection layer -- ssfunc1(wsel, p_min, omega_p) -- was removed, since the
# raw neighbour-difference product is no longer re-sharpened.
def test_shear_estimate_w_sel():
    nx = 64
    ny = 64
    scale = 0.2

    # pre-rendered (tests/data/ngmix_wsel.fits): a sheared Moffat PSF
    # (beta 2.5, fwhm 0.7) and an e = (0.1, -0.15) exponential (hlr 0.25,
    # slightly off-centre) under the shears (g1m/g1p/g2m/g2p = -/+0.02),
    # magnitudes (m0..m3 = 26.7, 26.8, 26.9, 27.0) and rotations (a0..a9,
    # fixed random angles; the test used to draw them unseeded) stepped
    # through below.  Keys as in make_fixtures.ngmix_wsel.
    fix = load("ngmix_wsel")
    psf_array = fix["psf"]
    n_mag = 4
    n_angle = 10

    # cell_overlap must be at least twice the background kernel reach
    # (2 * (3 arcsec / scale + 1) = 32 pixels here), otherwise the local
    # background would be estimated from pixels outside the cell.  Sizing the
    # cell as img + overlap keeps npatch = 1, so this is still a single cell
    # whose inner region is exactly the image.
    cell = anacal.geometry.get_cell_list(
        nx,
        ny,
        nx + 32,
        ny + 32,
        32,
        scale,
    )[0]

    cats = anacal.detector.find_peaks(
        img_array=fix["gal_init"],
        psf_array=psf_array,
        cell=cell,
        noise_array=None,
        image_bound=0,
        **kwargs,
    )
    assert len(cats) == 1
    assert cats[0].model.x1.v / scale == nx // 2
    assert cats[0].model.x2.v / scale == ny // 2

    # Test shear response calculation (no multiplicative bias)
    for i in range(n_mag):
        cat_1 = anacal.detector.find_peaks(
            img_array=fix[f"gal_m{i}_g1m"],
            psf_array=psf_array,
            cell=cell,
            noise_array=None,
            image_bound=0,
            **kwargs,
        )
        assert len(cat_1) == 1
        cat_1 = cat_1[0]

        cat_2 = anacal.detector.find_peaks(
            img_array=fix[f"gal_m{i}_g1p"],
            psf_array=psf_array,
            cell=cell,
            noise_array=None,
            image_bound=0,
            **kwargs,
        )
        assert len(cat_2) == 1
        cat_2 = cat_2[0]

        np.testing.assert_approx_equal(
            (cat_2.wsel.v - cat_1.wsel.v) / 0.04,
            (cat_2.wsel.g1 + cat_1.wsel.g1) / 2.0,
            2,
        )

        cat_1 = anacal.detector.find_peaks(
            img_array=fix[f"gal_m{i}_g2m"],
            psf_array=psf_array,
            cell=cell,
            noise_array=None,
            image_bound=0,
            **kwargs,
        )
        assert len(cat_1) == 1
        cat_1 = cat_1[0]

        cat_2 = anacal.detector.find_peaks(
            img_array=fix[f"gal_m{i}_g2p"],
            psf_array=psf_array,
            cell=cell,
            noise_array=None,
            image_bound=0,
            **kwargs,
        )
        assert len(cat_2) == 1
        cat_2 = cat_2[0]

        np.testing.assert_approx_equal(
            (cat_2.wsel.v - cat_1.wsel.v) / 0.04,
            (cat_2.wsel.g2 + cat_1.wsel.g2) / 2.0,
            2,
        )

    # Test symmetry (no additive bias)
    for i in range(n_angle):
        cat_1 = anacal.detector.find_peaks(
            img_array=fix[f"gal_a{i}_0"],
            psf_array=psf_array,
            cell=cell,
            noise_array=None,
            image_bound=0,
            **kwargs,
        )[0]

        cat_2 = anacal.detector.find_peaks(
            img_array=fix[f"gal_a{i}_90"],
            psf_array=psf_array,
            cell=cell,
            noise_array=None,
            image_bound=0,
            **kwargs,
        )[0]

        np.testing.assert_approx_equal(
            cat_1.wsel.v,
            cat_2.wsel.v,
            6,
        )
    return
