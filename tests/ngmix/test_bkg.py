import anacal
import numpy as np

from ..fixtures import load

# Same detector settings as test_wsel.
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

# Every image below is pre-rendered by tests/data/make_fixtures.py
# (ngmix_bkg): a sheared Moffat PSF (beta 2.5, fwhm PSF_FWHM) and, per
# configuration, one e = (0.1, -0.15) exponential at the image centre,
# optionally on a broad Gaussian pedestal.  The galaxies are float64
# because the shear step is DG.  The configuration tables below are
# duplicated in make_fixtures.py; the fixture key carries the position
# in the table, so keep the two in the same order.
FIX = load("ngmix_bkg")
PSF_ARRAY = FIX["psf"]


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
    # a mag 40 source on the mag 16 pedestal: the pedestal alone
    ped_only = FIX["ped_only"]
    level = float(np.median(ped_only[NY // 2 - 1:NY // 2 + 2,
                                     NX // 2 - 1:NX // 2 + 2]))
    # mag 22, hlr 0.3 on the same pedestal
    cat = _detect(FIX["ped_src"])
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

# (mag, hlr, comp): one detection at a fixed pixel, 0.02 < wsel < 0.98 at all
# three shear points, and the blend regime identical across the three.
WSEL_RESPONSE_CONFIGS = [
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


def _shear_triplet(prefix, mag, hlr, comp):
    """Catalogs at -DG, 0, +DG applied to ``comp``.

    The images are the fixtures ``<prefix>_m``, ``<prefix>_0`` and
    ``<prefix>_p``; mag/hlr/comp only label the assertion messages.

    Asserts the configuration is still usable -- exactly one detection, read at
    the same pixel in all three -- so that a code change that moves a fixed
    configuration out of its regime fails loudly instead of comparing numbers
    that are not comparable.
    """
    cats = []
    for tag, dv in (("m", -DG), ("0", 0.0), ("p", +DG)):
        c = _detect(FIX[f"{prefix}_{tag}"])
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
    for i, (mag, hlr) in enumerate(BKG_RESPONSE_CONFIGS):
        for comp in ("g1", "g2"):
            c1, _, c2 = _shear_triplet(f"bkg_{i}_{comp}", mag, hlr, comp)
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


def test_wsel_shear_response_with_background():
    """The background term's contribution to dwsel/dg.

    ``test_wsel`` uses a faint isolated source where bkg is ~0, so the
    background factor sits saturated at 1 and contributes nothing to the
    derivative.  Here the source is extended enough that ``data - bkg`` lands
    inside the cut's transition, so dwsel/dg genuinely depends on d(bkg)/dg.
    Measured agreement on the fixed configurations is 0.2% or better.
    """
    for i, (mag, hlr, comp) in enumerate(WSEL_RESPONSE_CONFIGS):
        c1, c0, c2 = _shear_triplet(f"wsel_{i}", mag, hlr, comp)
        for c in (c1, c0, c2):
            assert 0.02 < c.wsel.v < 0.98, (
                f"mag={mag} hlr={hlr} {comp}: wsel={c.wsel.v:.4f} left the "
                "sloped part of the cut; the fixed configuration has drifted"
            )
        fd = (c2.wsel.v - c1.wsel.v) / (2.0 * DG)
        an = 0.5 * (getattr(c1.wsel, comp) + getattr(c2.wsel, comp))
        scale = max(abs(fd), abs(an))
        assert scale > 1e-2, (
            f"dwsel/d{comp} vanished at mag={mag} hlr={hlr}; the fixed "
            "configuration no longer exercises the background term"
        )
        assert abs(fd - an) <= 0.005 * scale, (
            f"dwsel/d{comp} mismatch at mag={mag} hlr={hlr} "
            f"(wsel={c0.wsel.v:.4f}): step-by-hand {fd:.6g} vs "
            f"AnaCal {an:.6g} (differ by {abs(fd - an) / scale:.2%})"
        )
