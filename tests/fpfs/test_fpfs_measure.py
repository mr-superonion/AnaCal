import os

import anacal
import fitsio
import numpy as np
import numpy.lib.recfunctions as rfn

src_fname = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src_g.fits",
)

gal_fname = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gal_array.fits",
)

psf_fname = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "psf_array.fits",
)

br_fname = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bfunc_real.fits",
)

bi_fname = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bfunc_imag.fits",
)


def make_detection(gal_array, psf_array, scale, noise_variance):
    """Detect with the AnaCal detector (detector.h) and convert to the
    ('y', 'x') pixel catalogue FPFS measures at -- FPFS itself no longer
    detects.
    """
    det_task = anacal.task.Task(
        scale=scale,
        sigma_arcsec=0.53,
        snr_peak_min=10.0,
        omega_f=0.8,
        omega_v=0.04,
        mag_zero=30.0,
    )
    cells = anacal.geometry.get_cell_list(
        img_nx=gal_array.shape[1],
        img_ny=gal_array.shape[0],
        cell_nx=250,
        cell_ny=250,
        cell_overlap=80,
        scale=scale,
    )
    cat = det_task.process_image(
        np.asarray(gal_array, dtype=np.float32),
        psf_array,
        variance=noise_variance,
        cell_list=cells,
        do_measure=False,
    )
    det = np.zeros(len(cat), dtype=[("y", "f8"), ("x", "f8")])
    det["y"] = cat["x2_det"] / scale
    det["x"] = cat["x1_det"] / scale
    return det


def test_fpfs_measure():
    scale = 0.2
    ngrid2 = 64
    psf_array = fitsio.read(psf_fname)
    gal_array = fitsio.read(gal_fname)
    sigma_shapelets = 0.53
    std = 0.2

    ftask = anacal.fpfs.FpfsTask(
        npix=64,
        pixel_scale=scale,
        sigma_shapelets=sigma_shapelets,
        psf_array=psf_array,
    )

    det1 = make_detection(gal_array, psf_array, scale, std**2.0)
    det1 = det1[np.lexsort((det1["y"], det1["x"]))]
    src = ftask.run(
        gal_array=gal_array,
        psf=psf_array,
        det=det1,
    )
    assert src["noise"] is None
    src1 = rfn.structured_to_unstructured(src["data"])
    bfunc_real = fitsio.read(br_fname)
    bfunc_imag = fitsio.read(bi_fname)
    src2 = fitsio.read(src_fname)
    np.testing.assert_almost_equal(ftask.bfunc.real, bfunc_real)
    np.testing.assert_almost_equal(ftask.bfunc.imag, bfunc_imag)
    np.testing.assert_almost_equal(src1, src2, decimal=4)

    # use grid PSF
    psf_array2 = np.zeros((1, 1, ngrid2, ngrid2))
    psf_array2[0, 0] = psf_array
    # grid_psf = anacal.psf.GridPsf(
    #     x0=0,
    #     y0=0,
    #     dx=ngrid,
    #     dy=ngrid,
    #     model_array=psf_array2,
    # )

    # src = ftask.run(gal_array=gal_array, psf=grid_psf, det=det1)
    # src3 = rfn.structured_to_unstructured(src["data"])
    # np.testing.assert_almost_equal(src1, src3, decimal=5)
    return


if __name__ == "__main__":
    test_fpfs_measure()
