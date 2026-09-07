"""Render the images the AnaCal tests and examples use, once, with GalSim.

Run by hand from this directory::

    python make_fixtures.py

It writes one multi-extension FITS per test module into ``tests/data``
(plus the images the examples read), one HDU per image, keyed by
``EXTNAME``; the parameters that produced each image are in its header.
The tests read them through ``tests/fixtures.py`` and never import
GalSim, and this script is neither collected by pytest nor part of the
installed package.  Re-run it only when an image genuinely has to
change -- several tests compare against numbers derived from these
exact pixels.
"""

import datetime
import os
import subprocess

import cosmos_sim
import fitsio
import galsim
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.normpath(os.path.join(HERE, "..", "..", "examples", "fpfs"))


def _provenance():
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:  # pragma: no cover
        git = "unknown"
    return {
        "GSVER": galsim.__version__,
        "GENERAT": "tests/data/make_fixtures.py",
        "GITHASH": git,
        "DATE": datetime.date.today().isoformat(),
    }


class Fixture:
    """Collects (extname, array, header) and writes them to one file."""

    def __init__(self, path):
        self.path = path
        self.items = []

    def add(self, extname, array, **meta):
        assert len(extname) <= 68 and extname not in [
            it[0] for it in self.items
        ], extname
        hdr = dict(_provenance())
        for k, v in meta.items():
            assert len(k) <= 8, k
            hdr[k.upper()] = v if not isinstance(v, str) else v[:68]
        self.items.append((extname, np.ascontiguousarray(array), hdr))
        return array

    def write(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        with fitsio.FITS(self.path, "rw", clobber=True) as ff:
            for extname, arr, hdr in self.items:
                ff.write(arr, extname=extname, header=hdr)
        size = os.path.getsize(self.path) / 1e6
        print(
            f"{os.path.relpath(self.path, HERE):45s} "
            f"{len(self.items):3d} HDUs  {size:6.2f} MB"
        )


# --------------------------------------------------------------- helpers
def moffat(beta, fwhm, trunc=None, **shear):
    kw = {} if trunc is None else {"trunc": trunc}
    obj = galsim.Moffat(beta=beta, fwhm=fwhm, **kw)
    return obj.shear(**shear) if shear else obj


def draw_psf(psf_obj, n, scale, ny=None):
    """The standard PSF stamp: shifted by half a pixel, float32."""
    ny = n if ny is None else ny
    return (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=n, ny=ny, scale=scale)
        .array
    )


def draw_into(obj, nx, ny, scale, dtype=np.float32):
    """drawImage(image=..., add_to_image=True) into a fresh image."""
    if dtype is np.float32:
        img = galsim.ImageF(ncol=nx, nrow=ny, scale=scale)
    else:
        img = galsim.ImageD(ncol=nx, nrow=ny, scale=scale)
    obj.drawImage(image=img, add_to_image=True)
    return img.array


def cosmos(psf_obj, gname, seed, ny, nx, scale, nrot, buff=0, **kw):
    return cosmos_sim.make_isolated_sim(
        gal_type="mixed",
        sim_method="fft",
        psf_obj=psf_obj,
        gname=gname,
        seed=seed,
        ny=ny,
        nx=nx,
        scale=scale,
        do_shift=False,
        buff=buff,
        nrot_per_gal=nrot,
        **kw,
    )[0]


# ---------------------------------------------------------------- fpfs
def fpfs_process_image():
    fx = Fixture(os.path.join(HERE, "fpfs_process_image.fits"))
    scale, ngrid, nstamp, buff = 0.2, 64, 10, 15
    psf_obj = moffat(3.5, 0.6, trunc=0.6 * 4.0)
    fx.add(
        "psf",
        draw_psf(psf_obj, ngrid, scale),
        beta=3.5,
        fwhm=0.6,
        trunc=2.4,
        scale=scale,
    )
    fx.add(
        "gal",
        cosmos(
            psf_obj,
            "g1-1",
            2,
            ngrid * nstamp,
            ngrid * nstamp,
            scale,
            4,
            buff=buff,
            mag_zero=30,
        ),
        gname="g1-1",
        seed=2,
        nstamp=nstamp,
        ngrid=ngrid,
        buff=buff,
        nrot=4,
        magzero=30,
        scale=scale,
    )
    fx.write()


def fpfs_shearest():
    fx = Fixture(os.path.join(HERE, "fpfs_shearest.fits"))
    scale, ngrid, nrot = 0.2, 64, 12
    psf_obj = moffat(3.5, 0.6, trunc=0.6 * 4.0, e1=0.02, e2=-0.02)
    fx.add(
        "psf",
        draw_psf(psf_obj, ngrid, scale),
        beta=3.5,
        fwhm=0.6,
        trunc=2.4,
        e1=0.02,
        e2=-0.02,
        scale=scale,
    )
    for seed in (12, 23, 42):
        fx.add(
            f"gal_seed{seed}",
            cosmos(psf_obj, "g1-0", seed, ngrid, ngrid * nrot, scale, nrot),
            gname="g1-0",
            seed=seed,
            nrot=nrot,
            magzero=27,
            scale=scale,
        )
    fx.write()


FORCE_CASES = [
    (0.2, 0.0, 0.0),
    (0.2, 2.31, 0.43),
    (0.2, -2.35, 1.63),
    (0.164, -0.5, 1.5),
]


def fpfs_force_measure():
    fx = Fixture(os.path.join(HERE, "fpfs_force_measure.fits"))
    ngrid = 64
    for i, (scale, sx, sy) in enumerate(FORCE_CASES):
        psf_obj = moffat(3.5, 0.6, trunc=0.6 * 4.0, e1=0.02, e2=-0.02)
        fx.add(
            f"psf_{i}",
            draw_psf(psf_obj, ngrid, scale),
            beta=3.5,
            fwhm=0.6,
            trunc=2.4,
            e1=0.02,
            e2=-0.02,
            scale=scale,
        )
        gal = galsim.Gaussian(fwhm=0.6).shear(e1=0.2, e2=-0.24)
        fx.add(
            f"gal_{i}",
            gal.shift((0.5 + sx) * scale, (0.5 + sy) * scale)
            .drawImage(nx=ngrid, ny=ngrid, scale=scale)
            .array,
            fwhm=0.6,
            e1=0.2,
            e2=-0.24,
            shiftx=sx,
            shifty=sy,
            scale=scale,
        )
    fx.write()


def fpfs_noise_cov():
    fx = Fixture(os.path.join(HERE, "fpfs_noise_cov.fits"))
    fx.add(
        "psf_moffat35",
        draw_psf(moffat(3.5, 0.6, trunc=0.6 * 4.0, e1=0.02, e2=-0.02), 64, 0.2),
        beta=3.5,
        fwhm=0.6,
        trunc=2.4,
        e1=0.02,
        e2=-0.02,
        scale=0.2,
    )
    fx.add(
        "psf_moffat25",
        draw_psf(moffat(2.5, 0.8), 64, 0.2),
        beta=2.5,
        fwhm=0.8,
        scale=0.2,
    )
    fx.write()


# --------------------------------------------------------------- image
def image_noise_variance():
    fx = Fixture(os.path.join(HERE, "image_noise_variance.fits"))
    fx.add(
        "psf",
        draw_psf(moffat(2.5, 0.6, g1=0.02, g2=-0.02), 64, 0.2),
        beta=2.5,
        fwhm=0.6,
        g1=0.02,
        g2=-0.02,
        scale=0.2,
    )
    fx.write()


def image():
    fx = Fixture(os.path.join(HERE, "image.fits"))
    scale, ngrid = 0.2, 64
    psf_obj = moffat(3.5, 0.6, trunc=0.6 * 4.0, e1=0.02, e2=-0.02)
    fx.add(
        "psf_deconv",
        draw_psf(psf_obj, ngrid, scale),
        beta=3.5,
        fwhm=0.6,
        trunc=2.4,
        e1=0.02,
        e2=-0.02,
        scale=scale,
    )
    fx.add(
        "gal_deconv",
        cosmos(psf_obj, "g1-0", 1, ngrid, ngrid, scale, 1),
        gname="g1-0",
        seed=1,
        nrot=1,
        magzero=27,
        scale=scale,
    )
    obj = moffat(3.5, 0.6, trunc=3.0, e1=0.1, e2=-0.02)
    obj = obj.shift(4.5 * scale, 17 * scale)
    fx.add(
        "psf_rot",
        obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array,
        beta=3.5,
        fwhm=0.6,
        trunc=3.0,
        e1=0.1,
        e2=-0.02,
        offx=4.5,
        offy=17.0,
        scale=scale,
    )
    fx.add(
        "psf_rot90",
        obj.rotate(90 * galsim.degrees)
        .shift(0.5 * scale, 0.5 * scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=scale)
        .array,
        beta=3.5,
        fwhm=0.6,
        trunc=3.0,
        e1=0.1,
        e2=-0.02,
        offx=4.5,
        offy=17.0,
        rotdeg=90,
        scale=scale,
    )
    fx.write()


# --------------------------------------------------------------- ngmix
def ngmix_angle():
    fx = Fixture(os.path.join(HERE, "ngmix_angle.fits"))
    scale, n = 0.2, 64
    psf_obj = moffat(2.5, 0.7, g1=0.02, g2=-0.02)
    fx.add(
        "psf",
        draw_psf(psf_obj, n, scale),
        beta=2.5,
        fwhm=0.7,
        g1=0.02,
        g2=-0.02,
        scale=scale,
    )
    obj = galsim.Exponential(half_light_radius=0.16).shear(e1=0.2, e2=0)
    obj = obj.rotate(30.0 * galsim.degrees).withFlux(150.0)
    obj = galsim.Convolve(psf_obj, obj.shift(0.5 * scale, 0.5 * scale))
    fx.add(
        "gal",
        draw_into(obj, n, n, scale),
        hlr=0.16,
        e1=0.2,
        e2=0.0,
        angle=30.0,
        flux=150.0,
        scale=scale,
    )
    fx.write()


def ngmix_fpfs():
    fx = Fixture(os.path.join(HERE, "ngmix_fpfs.fits"))
    scale, n = 0.2, 64
    psf_obj = moffat(2.5, 0.8, g1=0.02, g2=-0.02)
    fx.add(
        "psf",
        draw_psf(psf_obj, n, scale),
        beta=2.5,
        fwhm=0.8,
        g1=0.02,
        g2=-0.02,
        scale=scale,
    )
    obj = galsim.Exponential(half_light_radius=0.21).shear(e1=0.2, e2=0)
    obj = obj.rotate(30.0 * galsim.degrees).withFlux(150.0)
    obj = galsim.Convolve(psf_obj, obj.shift(0.5 * scale, 0.5 * scale))
    fx.add(
        "gal",
        draw_into(obj, n, n, scale),
        hlr=0.21,
        e1=0.2,
        e2=0.0,
        angle=30.0,
        flux=150.0,
        scale=scale,
    )
    fx.write()


GAUSSFIT_DX = (-0.2, 0.11)


def ngmix_gaussfit():
    fx = Fixture(os.path.join(HERE, "ngmix_gaussfit.fits"))
    scale, n = 0.2, 64
    dx1, dx2 = GAUSSFIT_DX
    psf_obj = moffat(2.5, 0.7, g1=0.02, g2=-0.02)
    fx.add(
        "psf",
        draw_psf(psf_obj, n, scale),
        beta=2.5,
        fwhm=0.7,
        g1=0.02,
        g2=-0.02,
        scale=scale,
    )

    def sim(obj0, g1, g2, angle, flux, key, **meta):
        obj = obj0.rotate(angle * galsim.degrees).withFlux(flux)
        obj = obj.shear(g1=g1, g2=g2)
        obj = obj.shift((0.5 + dx1) * scale, (0.5 + dx2) * scale)
        obj = galsim.Convolve(psf_obj, obj)
        fx.add(
            key,
            draw_into(obj, n, n, scale),
            hlr=0.2,
            g1=g1,
            g2=g2,
            angle=angle,
            flux=flux,
            dx1=dx1,
            dx2=dx2,
            scale=scale,
            **meta,
        )

    # test_ngmix_gaussian_fit_additive: round exponential
    sim(
        galsim.Exponential(half_light_radius=0.2),
        0.0,
        0.0,
        0.0,
        150.0,
        "gal_add",
    )
    # test_ngmix_gaussian_fit2: intrinsically elliptical exponential
    obj0 = galsim.Exponential(half_light_radius=0.2).shear(e1=0.2, e2=-0.1)
    m = dict(e1=0.2, e2=-0.1)
    sim(obj0, 0.02, 0.0, 0.0, 150.0, "gal_g1p_f150", **m)
    sim(obj0, -0.02, 0.0, 0.0, 300.0, "gal_g1m_f300", **m)
    sim(obj0, -0.02, 0.0, 0.0, 150.0, "gal_g1m_f150", **m)
    sim(obj0, 0.0, 0.0, 0.0, 150.0, "gal_g0_a0", **m)
    sim(obj0, 0.0, 0.0, 90.0, 150.0, "gal_g0_a90", **m)
    # test_ngmix_gaussian_fit4: four sources on a 256 x 64 strip
    nx, ny = 256, 64
    fx.add(
        "psf_wide",
        draw_psf(psf_obj, nx, scale, ny=ny),
        beta=2.5,
        fwhm=0.7,
        g1=0.02,
        g2=-0.02,
        scale=scale,
    )
    obj = galsim.Gaussian(half_light_radius=0.25).shear(g1=0.03)
    obj = galsim.Convolve(psf_obj, obj)
    full = galsim.ImageF(ncol=nx, nrow=ny, scale=scale)
    centers = [(31.2, 31.2), (95.9, 32.05), (160, 32.1), (224, 31.8)]
    fluxes = [12, 23, 8.5, 18.4]
    for (cx, cy), flux in zip(centers, fluxes):
        shift = galsim.PositionD(
            (cx - (nx - 1) / 2) * scale, (cy - (ny - 1) / 2) * scale
        )
        obj.shift(shift).withFlux(flux).drawImage(image=full, add_to_image=True)
    fx.add(
        "gal_wide",
        full.array,
        hlr=0.25,
        g1=0.03,
        scale=scale,
        centers=str(centers),
        fluxes=str(fluxes),
    )
    fx.write()


WSEL_MAGS = np.arange(26.7, 27.1, 0.1)
WSEL_ANGLES = np.random.RandomState(0).random(10) * 360.0


def ngmix_wsel():
    fx = Fixture(os.path.join(HERE, "ngmix_wsel.fits"))
    scale, n = 0.2, 64
    psf_obj = moffat(2.5, 0.7, g1=0.02, g2=-0.02)
    fx.add(
        "psf",
        draw_psf(psf_obj, n, scale),
        beta=2.5,
        fwhm=0.7,
        g1=0.02,
        g2=-0.02,
        scale=scale,
    )
    obj0 = galsim.Exponential(half_light_radius=0.25)
    obj0 = obj0.shear(e1=0.1, e2=-0.15).shift(0.05 * scale, 0.1 * scale)

    def sim(key, g1, g2, angle=0.0, mag=26.8):
        obj = obj0.rotate(angle * galsim.degrees).shear(g1=g1, g2=g2)
        obj = obj.shift(0.5 * scale, 0.5 * scale)
        flux = 10 ** ((30.0 - mag) / 2.5)
        obj = galsim.Convolve(psf_obj, obj).withFlux(flux)
        fx.add(
            key,
            draw_into(obj, n, n, scale),
            hlr=0.25,
            e1=0.1,
            e2=-0.15,
            g1=g1,
            g2=g2,
            angle=float(angle),
            mag=float(mag),
            scale=scale,
        )

    sim("gal_init", -0.02, 0.0)
    for i, mag in enumerate(WSEL_MAGS):
        sim(f"gal_m{i}_g1m", -0.02, 0.0, mag=mag)
        sim(f"gal_m{i}_g1p", 0.02, 0.0, mag=mag)
        sim(f"gal_m{i}_g2m", 0.0, -0.02, mag=mag)
        sim(f"gal_m{i}_g2p", 0.0, 0.02, mag=mag)
    for i, ang in enumerate(WSEL_ANGLES):
        sim(f"gal_a{i}_0", 0.0, 0.0, angle=ang)
        sim(f"gal_a{i}_90", 0.0, 0.0, angle=ang + 90.0)
    fx.write()


BKG_DG = 0.002
BKG_RESPONSE_CONFIGS = [
    (25.8, 0.8),
    (25.8, 1.2),
    (26.0, 0.5),
    (26.0, 1.0),
    (26.5, 0.6),
    (26.8, 0.5),
]
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


def ngmix_bkg():
    fx = Fixture(os.path.join(HERE, "ngmix_bkg.fits"))
    scale, n = 0.2, 64
    psf_obj = moffat(2.5, 0.7, g1=0.02, g2=-0.02)
    fx.add(
        "psf",
        draw_psf(psf_obj, n, scale),
        beta=2.5,
        fwhm=0.7,
        g1=0.02,
        g2=-0.02,
        scale=scale,
    )

    def sim(key, g1, g2, mag, hlr, pedestal_mag=None):
        # float64 on purpose: the tests resolve a 0.002 shear step
        img = galsim.ImageD(ncol=n, nrow=n, scale=scale)
        obj = galsim.Exponential(half_light_radius=hlr)
        obj = obj.shear(e1=0.1, e2=-0.15).shear(g1=g1, g2=g2)
        obj = obj.withFlux(10.0 ** ((30.0 - mag) / 2.5))
        obj = obj.shift(0.5 * scale, 0.5 * scale)
        galsim.Convolve(psf_obj, obj).drawImage(image=img, add_to_image=True)
        meta = dict(
            hlr=hlr, e1=0.1, e2=-0.15, g1=g1, g2=g2, mag=mag, scale=scale
        )
        if pedestal_mag is not None:
            ped = galsim.Gaussian(sigma=12.0)
            ped = ped.withFlux(10.0 ** ((30.0 - pedestal_mag) / 2.5))
            ped = ped.shift(0.5 * scale, 0.5 * scale)
            galsim.Convolve(psf_obj, ped).drawImage(
                image=img, add_to_image=True
            )
            meta.update(pedsig=12.0, pedmag=pedestal_mag)
        fx.add(key, img.array, **meta)

    sim("ped_only", 0.0, 0.0, mag=40.0, hlr=0.3, pedestal_mag=16.0)
    sim("ped_src", 0.0, 0.0, mag=22.0, hlr=0.3, pedestal_mag=16.0)
    steps = (("m", -BKG_DG), ("0", 0.0), ("p", BKG_DG))
    for i, (mag, hlr) in enumerate(BKG_RESPONSE_CONFIGS):
        for comp in ("g1", "g2"):
            for tag, dv in steps:
                sh = {comp: dv, ("g2" if comp == "g1" else "g1"): 0.0}
                sim(f"bkg_{i}_{comp}_{tag}", sh["g1"], sh["g2"], mag, hlr)
    for i, (mag, hlr, comp) in enumerate(WSEL_RESPONSE_CONFIGS):
        for tag, dv in steps:
            sh = {comp: dv, ("g2" if comp == "g1" else "g1"): 0.0}
            sim(f"wsel_{i}_{tag}", sh["g1"], sh["g2"], mag, hlr)
    fx.write()


# ---------------------------------------------------------------- task
def task_detection():
    fx = Fixture(os.path.join(HERE, "task_detection.fits"))
    nn, scale, mag = 24, 0.2, 23.5
    psf_obj = moffat(2.5, 0.7, g1=0.02, g2=-0.02)
    fx.add(
        "psf",
        draw_psf(psf_obj, nn, scale),
        beta=2.5,
        fwhm=0.7,
        g1=0.02,
        g2=-0.02,
        scale=scale,
    )
    flux = 10 ** ((30.0 - mag) / 2.5)
    obj = galsim.Exponential(half_light_radius=0.30).shear(g1=0.03)
    obj = galsim.Convolve(psf_obj, obj.withFlux(flux))
    shift = galsim.PositionD(0.5 * scale, 0.5 * scale)
    fx.add(
        "stamp",
        draw_into(obj.shift(shift), nn, nn, scale),
        hlr=0.3,
        g1=0.03,
        mag=mag,
        scale=scale,
    )
    fx.write()


def task_flux_variance():
    fx = Fixture(os.path.join(HERE, "task_flux_variance.fits"))
    scale, npix = 0.2, 64
    psf_obj = moffat(3.5, 0.8, trunc=0.6 * 4.0, e1=0.02, e2=-0.02)
    fx.add(
        "psf",
        draw_psf(psf_obj, npix, scale),
        beta=3.5,
        fwhm=0.8,
        trunc=2.4,
        e1=0.02,
        e2=-0.02,
        scale=scale,
    )
    fx.add(
        "gal",
        cosmos(psf_obj, "g1-0", 0, npix, npix, scale, 1, mag_zero=30),
        gname="g1-0",
        seed=0,
        nrot=1,
        magzero=30,
        scale=scale,
    )
    fx.write()


MULTIBAND_PSFS = {"psf_07": 0.7, "psf_09": 0.9, "psf_06": 0.6}
MULTIBAND_STAMPS = {
    "stamp_07_235": (0.7, 23.5),
    "stamp_09_240": (0.9, 24.0),
    "stamp_06_245": (0.6, 24.5),
    "stamp_07_260": (0.7, 26.0),
}


def task_multiband():
    fx = Fixture(os.path.join(HERE, "task_multiband.fits"))
    scale, n = 0.2, 32
    for key, fwhm in MULTIBAND_PSFS.items():
        fx.add(
            key,
            draw_psf(moffat(2.5, fwhm, g1=0.02, g2=-0.02), n, scale),
            beta=2.5,
            fwhm=fwhm,
            g1=0.02,
            g2=-0.02,
            scale=scale,
        )
    for key, (fwhm, mag) in MULTIBAND_STAMPS.items():
        psf_obj = moffat(2.5, fwhm, g1=0.02, g2=-0.02)
        flux = 10 ** ((30.0 - mag) / 2.5)
        gal = galsim.Exponential(half_light_radius=0.30).shear(g1=0.03)
        gal = galsim.Convolve(psf_obj, gal.withFlux(flux))
        stamp = galsim.ImageF(ncol=n, nrow=n, scale=scale)
        gal.shift(galsim.PositionD(0.5 * scale, 0.5 * scale)).drawImage(
            image=stamp
        )
        fx.add(
            key,
            stamp.array,
            hlr=0.3,
            g1=0.03,
            mag=mag,
            psffwhm=fwhm,
            scale=scale,
        )
    fx.write()


# ----------------------------------------------------------------- psf
LANCZOS_ORDERS = (3, 4, 5, 6, 7, 9, 11)


def psf_lanczos():
    """galsim's Lanczos kernels, the reference for anacal.psf.lanczos_kernel.

    PIFF models are fit with galsim's interpolant, so the C++ kernel has
    to reproduce galsim (five aliasing terms in the conserve_dc
    correction), not the ideal kernel.  One HDU per order, columns:
    x, raw (conserve_dc=False), dc (conserve_dc=True).
    """
    fx = Fixture(os.path.join(HERE, "psf_lanczos.fits"))
    for n in LANCZOS_ORDERS:
        x = np.linspace(-n + 0.013, n - 0.021, 1777)
        raw = galsim.Lanczos(n, conserve_dc=False)
        dc = galsim.Lanczos(n, conserve_dc=True)
        table = np.stack(
            [
                x,
                [raw.xval(float(v)) for v in x],
                [dc.xval(float(v)) for v in x],
            ]
        ).astype(np.float64)
        fx.add(f"lanczos_{n}", table, order=n, rows="x, raw, dc")
    fx.write()


# ------------------------------------------------------------ examples
def examples_isolated():
    fx = Fixture(
        os.path.join(EXAMPLES, "isolated_galaxies", "isolated_sim.fits")
    )
    scale, ngrid, nstamp = 0.2, 64, 10
    psf_obj = moffat(3.5, 0.6, trunc=0.6 * 4.0)
    fx.add(
        "psf",
        draw_psf(psf_obj, ngrid, scale),
        beta=3.5,
        fwhm=0.6,
        trunc=2.4,
        scale=scale,
    )
    # example_fpfs_isolated.py: +/- shear fields of nstamp x nstamp
    for gname in ("g1-1", "g1-0"):
        fx.add(
            f"gal_{gname}",
            cosmos(
                psf_obj,
                gname,
                2,
                ngrid * nstamp,
                ngrid * nstamp,
                scale,
                4,
                mag_zero=30,
            ),
            gname=gname,
            seed=2,
            nstamp=nstamp,
            ngrid=ngrid,
            nrot=4,
            magzero=30,
            scale=scale,
        )
    # example_fpfs_isolated.ipynb: one galaxy and its 90-degree rotation
    fx.add(
        "gal_pair",
        cosmos(psf_obj, "g1-1", 2, ngrid, ngrid * 2, scale, 2, mag_zero=30),
        gname="g1-1",
        seed=2,
        nrot=2,
        magzero=30,
        scale=scale,
    )
    fx.write()


HIGHRES_SCENES = {
    # key: (stamp npix, n_multi_x, n_multi_y, nrot_per_gal)
    # The notebook used 10 x 10 and 50 x 50 grids (52 and 82 MB); the
    # stored scenes are the same set-up on fewer stamps.
    "scene_256": (256, 2, 2, 2),
    "scene_64": (64, 10, 10, 4),
}


def examples_isolated_highres():
    """Euclid-VIS-like scenes for example_fpfs_isolated_highres.ipynb."""
    fx = Fixture(
        os.path.join(EXAMPLES, "isolated_galaxies", "isolated_highres_sim.fits")
    )
    scale = 0.1
    psf_obj = moffat(2.5, 0.242)
    for n in (128, 64):
        fx.add(
            f"psf_{n}",
            draw_psf(psf_obj, n, scale),
            beta=2.5,
            fwhm=0.242,
            scale=scale,
        )
    for key, (npix, nmx, nmy, nrot) in HIGHRES_SCENES.items():
        fx.add(
            key,
            cosmos(
                psf_obj,
                "g1-1",
                1,
                npix * nmx,
                npix * 2 * nmy,
                scale,
                nrot,
                ngrid=npix,
                mag_zero=30,
            ),
            gname="g1-1",
            seed=1,
            ngrid=npix,
            nmultix=nmx,
            nmultiy=nmy,
            nrot=nrot,
            magzero=30,
            scale=scale,
        )
    fx.write()


def examples_covariance():
    fx = Fixture(os.path.join(EXAMPLES, "psf_moffat.fits"))
    fx.add(
        "psf",
        draw_psf(moffat(2.5, 0.8), 64, 0.2),
        beta=2.5,
        fwhm=0.8,
        scale=0.2,
    )
    fx.write()


if __name__ == "__main__":
    for fn in (
        fpfs_process_image,
        fpfs_shearest,
        fpfs_force_measure,
        fpfs_noise_cov,
        image_noise_variance,
        image,
        ngmix_angle,
        ngmix_fpfs,
        ngmix_gaussfit,
        ngmix_wsel,
        ngmix_bkg,
        task_detection,
        task_flux_variance,
        task_multiband,
        psf_lanczos,
        examples_isolated,
        examples_isolated_highres,
        examples_covariance,
    ):
        fn()
