import anacal
import galsim
import numpy as np

# Same detector settings as test_wdet.
kwargs = {
    "sigma_arcsec": 0.4,
    "snr_min": 10,
    "variance": 1.75e-2,
    "omega_f": 0.1,
    "omega_v": 0.010892341643103026,
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


def _block():
    # block_overlap must be at least twice the background kernel reach
    # (2 * (3 arcsec / scale + 1) = 32 pixels here).  Sizing the block as
    # img + overlap keeps npatch = 1, so this is a single block whose inner
    # region is exactly the image and whose centre is the image centre --
    # the latter matters because it makes decentralize() the identity, so the
    # g1/g2 slots are the plain shear derivatives.
    return anacal.geometry.get_block_list(
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
        block=_block(),
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


def _ring_ratios(arr, x_det, y_det):
    """The cascade's blend factors for this image, recomputed in numpy.

    Mirrors detector.h: four ring means about the detected pixel, then
    ``ratio_k = ssfunc1(running - b_k, 0, std_noise)``.  Used only to establish
    WHICH REGIME a configuration is in -- correctness is still judged by the
    C++ analytic derivative against a finite difference.  ssfunc1 is exactly 0
    or 1 (with zero derivative) outside +-std_noise, so a ratio strictly inside
    (0, 1) is what tells us the nonlinear branch is live.
    """
    blk = _block()
    blk.psf_array = PSF_ARRAY
    sigma_det = kwargs["sigma_arcsec"] * np.sqrt(2.0)
    data = np.asarray(
        anacal.image.prepare_data_block_image(
            img_array=np.ascontiguousarray(arr),
            psf_array=PSF_ARRAY,
            sigma_arcsec=sigma_det,
            block=blk,
        )[0]
    )
    std_noise = np.sqrt(
        anacal.image.get_smoothed_variance(
            SCALE, sigma_det, PSF_ARRAY, kwargs["variance"]
        )
    )
    nr = int(3.0 / SCALE) + 1
    dj, di = np.mgrid[-nr:nr + 1, -nr:nr + 1]
    r2 = (di * di + dj * dj) * SCALE * SCALE
    j = int(round(y_det / SCALE)) - blk.ymin
    i = int(round(x_det / SCALE)) - blk.xmin

    means = []
    for k in range(4):
        ra = 1.0 + 0.5 * k
        m = (r2 >= ra * ra) & (r2 < (ra + 0.5) ** 2)
        means.append(float(data[j + dj[m], i + di[m]].mean()))

    ratios, running = [], means[0]
    for k in range(1, 4):
        t = (running - means[k]) / (2.0 * std_noise) + 0.5
        rr = 0.0 if t < 0 else (1.0 if t > 1 else -2 * t ** 3 + 3 * t ** 2)
        ratios.append(rr)
        running = running + rr * (means[k] - running)
    return ratios


def _n_live(ratios, eps=0.02):
    return sum(1 for r in ratios if eps < r < 1.0 - eps)


def _fd_vs_analytic(mag, hlr, comp, rtol=0.002):
    """Compare AnaCal's d(bkg)/dg against the same slope worked out by hand.

    Only configurations where the smooth step is on its SLOPED part for all
    three of -DG, 0 and +DG are tested.  The step is flat outside +-std_noise
    and its slope there is exactly zero, so a shear step that begins on the flat
    part and ends on the sloped part is not following one smooth curve, and for
    a single galaxy the two numbers are then not expected to agree.  Anything
    else is skipped rather than asserted on.

    Returns None when the configuration is unusable.
    """
    other = "g2" if comp == "g1" else "g1"
    shears = [{comp: -DG, other: 0.0},
              {comp: 0.0, other: 0.0},
              {comp: +DG, other: 0.0}]
    images, cats = [], []
    for sh in shears:
        im = _draw(mag=mag, hlr=hlr, **sh)
        c = _detect(im)
        if len(c) != 1:
            return None
        images.append(im)
        cats.append(c[0])
    # bkg must be read at the same pixel in all three, or the difference mixes
    # in a change of position
    if len({(c.x1_det, c.x2_det) for c in cats}) != 1:
        return None

    live = [_n_live(_ring_ratios(im, c.x1_det, c.x2_det))
            for im, c in zip(images, cats)]
    if not all(n >= 1 for n in live):
        return None

    c1, c2 = cats[0], cats[2]
    fd = (c2.bkg.v - c1.bkg.v) / (2.0 * DG)
    an = 0.5 * (getattr(c1.bkg, comp) + getattr(c2.bkg, comp))
    scale = max(abs(fd), abs(an))
    if scale < 1e-4:
        return None
    assert abs(fd - an) <= rtol * scale, (
        f"d(bkg)/d{comp} mismatch at mag={mag} hlr={hlr} (live ratios "
        f"{live}): step-by-hand {fd:.6g} vs AnaCal {an:.6g} "
        f"(differ by {abs(fd - an) / scale:.2%})"
    )
    return abs(fd - an) / scale


def test_bkg_shear_response():
    """d(bkg)/dg, on the sloped part of the smooth step.

    The blend factors come within std_noise of each other -- putting the step on
    its sloped part -- for FAINT compact sources near the detection limit, which
    is the grid below.  There the finite difference converges as DG^2 to about
    5e-6 relative, so the tolerance can be tight.

    If these magnitudes ever stop being detected, or stop landing on the sloped
    part, the assertion at the end fires instead of the test quietly checking
    nothing.
    """
    tested, worst = 0, 0.0
    for mag in (25.8, 26.0, 26.2, 26.5):
        for hlr in (0.6, 0.8, 1.0):
            for comp in ("g1", "g2"):
                e = _fd_vs_analytic(mag, hlr, comp)
                if e is not None:
                    tested += 1
                    worst = max(worst, e)
    assert tested >= 4, (
        f"only {tested} configurations had all three shears on the sloped part "
        "of the smooth step; the test would check nothing"
    )
    print(f"bkg shear response: {tested} points, worst {worst:.3%}")


def test_wdet_shear_response_with_background():
    """The background term's contribution to dwdet/dg.

    ``test_wdet`` uses a faint isolated source where bkg is ~0, so the
    background factor sits saturated at 1 and contributes nothing to the
    derivative.  Here the source is extended enough that ``data - bkg`` lands
    inside the cut's transition, so dwdet/dg genuinely depends on d(bkg)/dg.
    """
    tested = 0
    for mag in (23.0, 23.5, 24.0, 24.5):
        for hlr in (1.5, 2.0, 2.5):
            for comp in ("g1", "g2"):
                other = "g2" if comp == "g1" else "g1"
                cats = []
                for dv in (-DG, 0.0, +DG):
                    sh = {comp: dv, other: 0.0}
                    c = _detect(_draw(mag=mag, hlr=hlr, **sh))
                    if len(c) != 1:
                        break
                    cats.append(c[0])
                if len(cats) != 3:
                    continue
                # wdet must be on the sloped part of the cut for all three, and
                # read at the same pixel, or the two numbers are not comparable
                if not all(0.02 < c.wdet.v < 0.98 for c in cats):
                    continue
                if len({(c.x1_det, c.x2_det) for c in cats}) != 1:
                    continue
                c1, c2 = cats[0], cats[2]
                fd = (c2.wdet.v - c1.wdet.v) / (2.0 * DG)
                an = 0.5 * (getattr(c1.wdet, comp) + getattr(c2.wdet, comp))
                scale = max(abs(fd), abs(an))
                if scale < 1e-2:
                    continue
                tested += 1
                assert abs(fd - an) <= 0.1 * scale, (
                    f"dwdet/d{comp} mismatch at mag={mag} hlr={hlr} "
                    f"(wdet={cats[1].wdet.v:.4f}): step-by-hand {fd:.6g} vs "
                    f"AnaCal {an:.6g}"
                )
    assert tested >= 2, (
        f"only {tested} configurations put the background cut in transition; "
        "the test would pass vacuously"
    )
