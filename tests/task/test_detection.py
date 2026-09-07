import gc

import anacal
import numpy as np
from memory_profiler import memory_usage

from .. import mem_used, print_mem
from ..fixtures import load


def test_task_detection():
    nn = 24
    scale = 0.2
    ngal = 20
    # pre-rendered (tests/data/task_detection.fits): a sheared Moffat PSF
    # (beta 2.5, fwhm 0.7) and one g1 = 0.03 exponential (hlr 0.3, mag
    # 23.5) centred on an nn x nn stamp; the field is that stamp tiled
    fix = load("task_detection")
    psf_array = fix["psf"]
    img_array = np.tile(fix["stamp"], (ngal, ngal))
    kwargs = {
        "omega_f": 0.8,
        "omega_v": 0.04,
    }
    noise_variance = 0.10
    prior = anacal.ngmix.modelPrior()
    prior.set_sigma_x(anacal.math.qnumber(0.5))
    det_task = anacal.task.Task(
        scale=scale,
        sigma_arcsec=0.4,
        snr_peak_min=10,
        stamp_size=nn,
        image_bound=0,
        num_epochs=30,
        prior=prior,
        force_size=True,
        **kwargs,
    )
    cells = anacal.geometry.get_cell_list(
        img_array.shape[0],
        img_array.shape[1],
        500,
        500,
        64,
        scale,
    )

    initial_memory_usage = mem_used()
    print_mem(initial_memory_usage)

    def func():
        det_task.process_image(
            img_array,
            psf_array,
            variance=noise_variance,
            cell_list=cells,
        )
        gc.collect()
        return

    peak_memory_usage = max(memory_usage(proc=(func,)))
    print("Peak Mem:", peak_memory_usage, "M")

    final_memory_usage = mem_used()
    print("Additional Mem:")
    print_mem(final_memory_usage - initial_memory_usage)

    catalog = det_task.process_image(
        img_array, psf_array, variance=noise_variance, cell_list=cells,
    )
    assert len(catalog) == ngal * ngal

    ind = np.lexsort(
        (np.round(catalog["x1"] / scale), np.round(catalog["x2"] / scale)),
    )
    crange = np.arange(nn // 2, img_array.shape[0], nn)
    centers = np.array([[x, y] for y in crange for x in crange])
    np.testing.assert_allclose(
        centers[:, 0] - np.round(catalog["x1"][ind] / 0.2),
        0.0,
    )
    np.testing.assert_allclose(
        centers[:, 1] - np.round(catalog["x2"][ind] / 0.2),
        0.0,
    )

    cells = anacal.geometry.get_cell_list(
        img_array.shape[0],
        img_array.shape[1],
        512,
        512,
        150,
        scale,
    )
    catalog = det_task.process_image(
        img_array,
        psf_array,
        variance=noise_variance,
        cell_list=cells,
    )

    assert len(catalog) == ngal*ngal
    ind = np.lexsort(
        (np.round(catalog["x1"] / scale), np.round(catalog["x2"] / scale))
    )
    crange = np.arange(nn // 2, img_array.shape[0], nn)
    centers = np.array([[x, y] for y in crange for x in crange])
    np.testing.assert_allclose(
        centers[:, 0],
        np.round(catalog["x1"][ind] / 0.2),
    )
    np.testing.assert_allclose(
        centers[:, 1],
        np.round(catalog["x2"][ind] / 0.2),
    )

    e1 = catalog["fpfs_e1"] * catalog["wsel"]
    r1 = (
        catalog["fpfs_de1_dg1"] * catalog["wsel"]
        + catalog["dwsel_dg1"] * catalog["fpfs_e1"]
    )
    print(np.sum(e1) / np.sum(r1))
    return
