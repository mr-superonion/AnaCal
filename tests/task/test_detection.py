import gc

import anacal
import galsim
import numpy as np
from memory_profiler import memory_usage

from .. import mem_used, print_mem


def test_task_detection():
    nn = 24
    mag = 23.5

    flux = 10 ** ((30.0 - mag) / 2.5)

    scale = 0.2
    psf_fwhm = 0.7
    ngal = 20
    # PSF
    psf_obj = galsim.Moffat(
        beta=2.5,
        fwhm=psf_fwhm,
    ).shear(
        g1=0.02,
        g2=-0.02,
    )
    psf_array = (
        psf_obj.shift(0.5 * scale, 0.5 * scale)
        .drawImage(
            nx=nn,
            ny=nn,
            scale=scale,
        )
        .array
    )

    obj = galsim.Exponential(
        half_light_radius=0.30
    ).shear(g1=0.03).withFlux(flux)
    obj = galsim.Convolve(psf_obj, obj)

    # Create an empty image
    full_image = galsim.ImageF(ncol=nn, nrow=nn, scale=scale)

    # Define centers
    crange = np.arange(nn // 2, nn, nn)
    centers = [(x, y) for x in crange for y in crange]

    # Draw galaxies at specified positions
    for center in centers:
        shift = galsim.PositionD(
            (center[0] - nn / 2 + 0.5) * scale,
            (center[1] - nn / 2 + 0.5) * scale,
        )
        final_galaxy = obj.shift(shift)
        final_galaxy.drawImage(image=full_image, add_to_image=True)
    img_array = np.tile(full_image.array, (ngal, ngal))
    kwargs = {
        "omega_f": 0.8,
        "v_min": 0.020,
        "omega_v": 0.04,
        "p_min": 0.15,
        "omega_p": 0.05,
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
    blocks = anacal.geometry.get_block_list(
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
            block_list=blocks,
        )
        gc.collect()
        return

    peak_memory_usage = max(memory_usage(proc=(func,)))
    print("Peak Mem:", peak_memory_usage, "M")

    final_memory_usage = mem_used()
    print("Additional Mem:")
    print_mem(final_memory_usage - initial_memory_usage)

    catalog = det_task.process_image(
        img_array, psf_array, variance=noise_variance, block_list=blocks,
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

    blocks = anacal.geometry.get_block_list(
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
        block_list=blocks,
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
