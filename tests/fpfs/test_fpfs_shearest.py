import anacal
import numpy as np
import pytest

from ..fixtures import load

ngrid = 64
mag_zero = 27

# tests/data/fpfs_shearest.fits: a sheared Moffat PSF and, per seed, one
# row of 12 rotated copies of a COSMOS galaxy sheared by g1 = -0.02
# ("g1-0"), 0.2 arcsec/pixel, mag_zero 27
FIX = load("fpfs_shearest")


def simulate_gal_psf(scale, seed, rcut, gcomp="g1", nrot=4):
    assert (scale, gcomp, nrot) == (0.2, "g1", 12), "not pre-rendered"
    psf_array = FIX["psf"][
        ngrid // 2 - rcut : ngrid // 2 + rcut,
        ngrid // 2 - rcut : ngrid // 2 + rcut,
    ]
    gal_array = FIX[f"gal_seed{seed}"]

    # force detection at center
    indx = np.arange(ngrid // 2, ngrid * nrot, ngrid)
    indy = np.arange(ngrid // 2, ngrid, ngrid)
    inds = np.meshgrid(indy, indx, indexing="ij")
    coords = np.vstack(inds).T
    coords = [(cc[0], cc[1]) for cc in coords]
    return gal_array, psf_array, coords


def do_test(scale, seed, rcut, gcomp):
    if gcomp == "g1":
        g1 = -0.02
        g2 = 0.0
    elif gcomp == "g2":
        g1 = 0.0
        g2 = -0.02
    else:
        raise ValueError("gcomp should be g1 or g2")
    sigma_shapelets = 0.53

    nrot = 12
    gal_array, psf_array, coords = simulate_gal_psf(
        scale, seed, rcut, gcomp, nrot
    )

    # shape_order=4: measure_fpfs takes the 12-mode FpfsShapelets record
    # and the 4th-order estimator q is checked below
    ftask = anacal.fpfs.FpfsTask(
        npix=64,
        pixel_scale=scale,
        sigma_shapelets=sigma_shapelets,
        psf_array=psf_array,
        shape_order=4,
    )

    src = ftask.run(
        gal_array=gal_array,
        psf=psf_array,
        det=coords,
    )

    ells = anacal.fpfs.measure_fpfs(
        C0=4,
        x_array=src["data"],
        y_array=src["noise"],
    )

    # The 2nd order shear estimator
    g1_est = np.average(ells["e1"]) / np.average(ells["de1_dg1"])
    g2_est = np.average(ells["e2"]) / np.average(ells["de2_dg2"])
    assert np.abs(g1_est - g1) < 3e-5
    assert np.abs(g2_est - g2) < 3e-5

    # The 4th order shear estimator
    g1_est = np.average(ells["q1"]) / np.average(ells["dq1_dg1"])
    g2_est = np.average(ells["q2"]) / np.average(ells["dq2_dg2"])
    assert np.abs(g1_est - g1) < 3e-5
    assert np.abs(g2_est - g2) < 3e-5

    # run as a list
    gal_list = [gal_array[:, i * ngrid : (i + 1) * ngrid] for i in range(nrot)]
    psf_list = [psf_array] * nrot
    src_g = np.array(
        [
            ftask.run(gal_array=gal_list[i], psf=psf_list[i])["data"][0]
            for i in range(nrot)
        ],
        dtype=src["data"].dtype,
    )
    assert np.all(src["data"] == src_g)
    return


@pytest.mark.parametrize("seed", [12, 23, 42])
def test_shear_estimation(seed):
    do_test(0.2, seed, 32, "g1")
    return


def _fit_catalog(gal_array, psf_array, coords, scale, sigma_shapelets):
    """The detector's Gaussian fit at the given centres (force_center):
    the intrinsic covariance and its shear response, which set the
    adaptive kernel's size."""
    nrot = len(coords)
    x1 = np.array([c[1] for c in coords], dtype=np.float64) * scale
    x2 = np.array([c[0] for c in coords], dtype=np.float64) * scale
    det = anacal.table.make_catalog_empty(x1, x2)
    det["x1"], det["x2"] = x1, x2
    det["x1_det"], det["x2_det"] = x1, x2
    task = anacal.task.Task(
        scale=scale,
        sigma_arcsec=sigma_shapelets / np.sqrt(2.0),
        snr_peak_min=5.0,
        omega_f=0.218,
        omega_v=0.011,
        stamp_size=ngrid,
        num_epochs=20,
        force_center=True,
        fpfs_c0=0.0,
    )
    cells = anacal.geometry.get_cell_list(
        img_nx=gal_array.shape[1], img_ny=gal_array.shape[0],
        cell_nx=ngrid, cell_ny=ngrid, cell_overlap=0, scale=scale,
    )
    cat = task.process_image(
        np.asarray(gal_array, dtype=np.float32), psf_array,
        variance=1e-4, cell_list=cells, detection=det,
    )
    assert len(cat) == nrot
    cat = cat[np.argsort(cat["x1_det"])]
    out = np.zeros(
        nrot,
        dtype=[("y", "f8"), ("x", "f8")]
        + [(c, "f8") for c in anacal.fpfs.ADAPTIVE_FIT_COLUMNS],
    )
    out["y"] = cat["x2_det"] / scale
    out["x"] = cat["x1_det"] / scale
    for c in anacal.fpfs.ADAPTIVE_FIT_COLUMNS:
        out[c] = cat[c]
    return out


@pytest.mark.parametrize("seed", [12, 23, 42])
def test_shear_estimation_adaptive(seed):
    """The ring test on the adaptive kernels afpfs1 and afpfs2 (kernel
    ratio 1.5): the same 12 rotated copies, each measured with its own
    kernel size from the Gaussian fit.  The size changes with the
    shear, and only with that term in the response does the estimator
    meet the fixed kernel's accuracy."""
    scale, rcut, g1 = 0.2, 32, -0.02
    sigma_shapelets = 0.53
    gal_array, psf_array, coords = simulate_gal_psf(
        scale, seed, rcut, "g1", 12
    )
    det = _fit_catalog(gal_array, psf_array, coords, scale, sigma_shapelets)
    cfg = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=sigma_shapelets, adaptive_sigma=True,
        adaptive_fpfs_kernel_ratio=1.5, shape_order=4, c0=4.0,
    )
    kw = dict(
        fpfs_config=cfg, mag_zero=mag_zero, gal_array=gal_array,
        psf_array=psf_array, pixel_scale=scale, noise_variance=1e-4,
    )
    out = anacal.fpfs.process_image(detection=det, **kw)
    # Without the size's shear response the ring test must fail
    det0 = det.copy()
    for c in ("dmxx_dg1", "dmxx_dg2", "dmyy_dg1", "dmyy_dg2"):
        det0[c] = 0.0
    out0 = anacal.fpfs.process_image(detection=det0, **kw)
    assert np.all(out["afpfs2_sigma"] > out["afpfs1_sigma"])
    for k in ("afpfs1", "afpfs2"):
        assert np.all(out[f"{k}_sigma"] > sigma_shapelets)
        assert np.any(out[f"{k}_dsigma_dg1"] != 0.0)
        # The 2nd and 4th order shear estimators
        g1_e = np.average(out[f"{k}_e1"]) / np.average(out[f"{k}_de1_dg1"])
        g1_q = np.average(out[f"{k}_q1"]) / np.average(out[f"{k}_dq1_dg1"])
        assert np.abs(g1_e - g1) < 3e-5, k
        assert np.abs(g1_q - g1) < 3e-5, k
        assert np.all(out0[f"{k}_dsigma_dg1"] == 0.0)
        g1_0 = np.average(out0[f"{k}_e1"]) / np.average(out0[f"{k}_de1_dg1"])
        assert np.abs(g1_0 - g1) > 3e-5, k
    return
