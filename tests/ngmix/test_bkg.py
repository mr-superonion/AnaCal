import anacal
import galsim
import numpy as np

# Same detector settings as test_wdet.
kwargs = {
    "sigma_arcsec": 0.4,
    "snr_min": 10,
    "variance": 1.75e-2,
    "omega_f": 0.1,
    "omega_v": 0.011,
}

NX = NY = 64
SCALE = 0.2
PSF_FWHM = 0.7
# Shear step for the central difference.  With float64 rendering the truncation
# error falls as DG^2 and only turns over into round-off well below 1e-4, so
# 0.002 sits comfortably in the converged region: measured agreement in the
# blended-ring regime is 0.0012% here, versus 0.075% at DG = 0.02.
DG = 0.002

psf_obj = galsim.Moffat(beta=2.5, fwhm=PSF_FWHM).shear(g1=0.02, g2=-0.02)
PSF_ARRAY = (
    psf_obj.shift(0.5 * SCALE, 0.5 * SCALE)
    .drawImage(nx=NX, ny=NY, scale=SCALE)
    .array
)


def _cell():
    # cell_overlap must be at least twice the background kernel reach
    # (2 * (3 arcsec / scale + 1) = 32 pixels here).  Sizing the cell as
    # img + overlap keeps npatch = 1, so this is a single cell whose inner
    # region is exactly the image and whose centre is the image centre --
    # the latter matters because it makes decentralize() the identity, so the
    # g1/g2 slots are the plain shear derivatives.
    return anacal.geometry.get_cell_list(
        NX, NY, NX + 32, NY + 32, 32, SCALE
    )[0]


def _draw(g1, g2, mag, hlr, pedestal_mag=None):
    """One extended source at the image centre, optionally on a flat-ish
    pedestal.  The source is extended so that its own light reaches the 1-3
    arcsec background rings, which is what makes ``bkg`` respond to shear."""
    # ImageD (float64): rendering at single precision puts a round-off floor
    # on how small a shear step the finite difference can resolve.
    img = galsim.ImageD(ncol=NX, nrow=NY, scale=SCALE)
    obj = galsim.Exponential(half_light_radius=hlr)
    obj = obj.shear(e1=0.1, e2=-0.15).shear(g1=g1, g2=g2)
    obj = obj.withFlux(10.0 ** ((30.0 - mag) / 2.5))
    obj = obj.shift(0.5 * SCALE, 0.5 * SCALE)
    galsim.Convolve(psf_obj, obj).drawImage(image=img, add_to_image=True)
    if pedestal_mag is not None:
        ped = galsim.Gaussian(sigma=12.0)
        ped = ped.withFlux(10.0 ** ((30.0 - pedestal_mag) / 2.5))
        ped = ped.shift(0.5 * SCALE, 0.5 * SCALE)
        galsim.Convolve(psf_obj, ped).drawImage(image=img, add_to_image=True)
    return img.array


def _detect(arr):
    return anacal.detector.find_peaks(
        img_array=arr,
        psf_array=PSF_ARRAY,
        cell=_cell(),
        noise_array=None,
        image_bound=0,
        **kwargs,
    )



def test_bkg_tracks_a_flat_background():
    """On a broad pedestal the estimate should recover the local level.

    The pedestal is wide compared with the 3 arcsec ring span, so within the
    rings it is nearly flat and ``bkg`` must sit close to the pedestal surface
    brightness at the source position.
    """
    ped_only = _draw(0.0, 0.0, mag=40.0, hlr=0.3, pedestal_mag=16.0)
    level = float(np.median(ped_only[NY // 2 - 1:NY // 2 + 2,
                                     NX // 2 - 1:NX // 2 + 2]))
    cat = _detect(_draw(0.0, 0.0, mag=22.0, hlr=0.3, pedestal_mag=16.0))
    assert len(cat) == 1
    # a compact source barely leaks into the rings, so this is the pedestal
    np.testing.assert_allclose(cat[0].bkg.v, level, rtol=0.2)


# Configurations for the shear-response tests below.  They are FIXED: the
# region selection was done offline (recomputing the cascade's blend ratios in
# numpy for a grid of mag/hlr) and only configurations where -DG, 0 and +DG
# all keep the smooth step in the same regime were kept -- a shear step that
# starts on a flat branch and ends on the sloped one does not follow a single
# smooth curve, and the finite difference then need not match the analytic
# slope.  The tests themselves only draw, measure and compare.
#
# (mag, hlr): the blend is on its SLOPED part at all three shear points for
# both components.
BKG_RESPONSE_CONFIGS = [
    (25.8, 0.8),
    (25.8, 1.2),
    (26.0, 0.5),
    (26.0, 1.0),
    (26.5, 0.6),
    (26.8, 0.5),
]

# (mag, hlr, comp): one detection at a fixed pixel, 0.02 < wdet < 0.98 at all
# three shear points, and the blend regime identical across the three.
WDET_RESPONSE_CONFIGS = [
    (23.0, 2.5, "g1"),
    (23.0, 2.5, "g2"),
    (23.5, 2.0, "g1"),
    (23.5, 2.0, "g2"),
    (24.0, 1.5, "g1"),
    (24.0, 1.5, "g2"),
    (24.0, 2.0, "g1"),
    (24.0, 2.0, "g2"),
    (24.5, 1.5, "g1"),
]


def _shear_triplet(mag, hlr, comp):
    """Catalogs at -DG, 0, +DG applied to ``comp``.

    Asserts the configuration is still usable -- exactly one detection, read at
    the same pixel in all three -- so that a code change that moves a fixed
    configuration out of its regime fails loudly instead of comparing numbers
    that are not comparable.
    """
    other = "g2" if comp == "g1" else "g1"
    cats = []
    for dv in (-DG, 0.0, +DG):
        sh = {comp: dv, other: 0.0}
        c = _detect(_draw(mag=mag, hlr=hlr, **sh))
        assert len(c) == 1, (
            f"mag={mag} hlr={hlr} {comp}={dv}: expected one detection, "
            f"got {len(c)}; the fixed test configuration has drifted"
        )
        cats.append(c[0])
    assert len({(c.x1_det, c.x2_det) for c in cats}) == 1, (
        f"mag={mag} hlr={hlr} {comp}: detected pixel moved with the shear "
        "step; the finite difference would mix in a change of position"
    )
    return cats


def test_bkg_shear_response():
    """d(bkg)/dg against a central finite difference.

    On the fixed configurations the finite difference converges as DG^2 to
    about 5e-6 relative, and the measured agreement is 0.06% or better, so the
    tolerance can be tight.
    """
    for mag, hlr in BKG_RESPONSE_CONFIGS:
        for comp in ("g1", "g2"):
            c1, _, c2 = _shear_triplet(mag, hlr, comp)
            fd = (c2.bkg.v - c1.bkg.v) / (2.0 * DG)
            an = 0.5 * (getattr(c1.bkg, comp) + getattr(c2.bkg, comp))
            scale = max(abs(fd), abs(an))
            assert scale > 1e-4, (
                f"d(bkg)/d{comp} vanished at mag={mag} hlr={hlr}; the fixed "
                "configuration no longer exercises the response"
            )
            assert abs(fd - an) <= 0.002 * scale, (
                f"d(bkg)/d{comp} mismatch at mag={mag} hlr={hlr}: "
                f"step-by-hand {fd:.6g} vs AnaCal {an:.6g} "
                f"(differ by {abs(fd - an) / scale:.2%})"
            )


def test_wdet_shear_response_with_background():
    """The background term's contribution to dwdet/dg.

    ``test_wdet`` uses a faint isolated source where bkg is ~0, so the
    background factor sits saturated at 1 and contributes nothing to the
    derivative.  Here the source is extended enough that ``data - bkg`` lands
    inside the cut's transition, so dwdet/dg genuinely depends on d(bkg)/dg.
    Measured agreement on the fixed configurations is 0.2% or better.
    """
    for mag, hlr, comp in WDET_RESPONSE_CONFIGS:
        c1, c0, c2 = _shear_triplet(mag, hlr, comp)
        for c in (c1, c0, c2):
            assert 0.02 < c.wdet.v < 0.98, (
                f"mag={mag} hlr={hlr} {comp}: wdet={c.wdet.v:.4f} left the "
                "sloped part of the cut; the fixed configuration has drifted"
            )
        fd = (c2.wdet.v - c1.wdet.v) / (2.0 * DG)
        an = 0.5 * (getattr(c1.wdet, comp) + getattr(c2.wdet, comp))
        scale = max(abs(fd), abs(an))
        assert scale > 1e-2, (
            f"dwdet/d{comp} vanished at mag={mag} hlr={hlr}; the fixed "
            "configuration no longer exercises the background term"
        )
        assert abs(fd - an) <= 0.005 * scale, (
            f"dwdet/d{comp} mismatch at mag={mag} hlr={hlr} "
            f"(wdet={c0.wdet.v:.4f}): step-by-hand {fd:.6g} vs "
            f"AnaCal {an:.6g} (differ by {abs(fd - an) / scale:.2%})"
        )
