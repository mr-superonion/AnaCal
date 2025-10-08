import os

import anacal
import galsim
import numpy as np

data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../examples/fpfs/blended_galaxies",
)


def gaussian_flux_variance(
    img_array,
    psf_array,
    sigma_arcsec,
    sigma_kernel,
    pixel_scale=1.0,
    noise_variance=1.0,
    eps=1e-6,
    noise_corr=None,
):
    ny, nx = psf_array.shape
    fx = np.fft.fftfreq(nx, d=1.0)
    fy = np.fft.fftfreq(ny, d=1.0)
    kx, ky = np.meshgrid(2*np.pi*fx, 2*np.pi*fy)
    k2 = kx**2 + ky**2
    psf_array = psf_array / psf_array.sum()
    P = np.fft.fft2(np.fft.ifftshift(psf_array))
    sigma_pix_sm = sigma_arcsec / pixel_scale
    sigma_pix_fit = np.sqrt(sigma_arcsec**2.0 + sigma_kernel**2.0) / pixel_scale
    T = np.exp(-0.5 * sigma_pix_sm**2 * k2)
    denom = P.copy()
    tiny = eps * np.abs(P[0, 0])
    denom[np.abs(denom) < tiny] = tiny
    H = T / denom
    W = np.exp(-0.5 * sigma_pix_fit**2 * k2)
    ff = 4.0 * np.pi * sigma_pix_fit**2
    if noise_corr is not None:
        noise_corr = np.pad(noise_corr * noise_variance, (8, 7))
        noise_pow = np.fft.fft2(np.fft.ifftshift(noise_corr)).real
    else:
        noise_pow = noise_variance
    var_flux = np.sum(np.abs(W * H)**2 * noise_pow) * (ff**2) / (nx * ny)
    img_f = np.fft.fft2(np.fft.ifftshift(img_array))
    flux = np.sum(np.conj(W*H) * img_f).real * ff / (nx * ny)
    return flux, var_flux


def test_flux_variance():
    mag_zero = 30
    noise_std = 0.3
    pixel_scale = 0.2
    sigma_arcsec = 0.38
    sigma_shapelets = sigma_arcsec * np.sqrt(2.0)
    npix = 64
    psf_obj = galsim.Moffat(beta=3.5, fwhm=0.8, trunc=0.6 * 4.0).shear(
        e1=0.02, e2=-0.02
    )

    psf_array = (
        psf_obj.shift(0.5 * pixel_scale, 0.5 * pixel_scale)
        .drawImage(nx=npix, ny=npix, scale=pixel_scale)
        .array
    )
    psf_array = np.asarray(
        anacal.utils.resize_array(
            psf_array, (npix, npix)
        ),
        dtype=np.float64,
    )
    gname = "g1-0"
    gal_array = anacal.simulation.make_isolated_sim(
        gal_type="mixed",
        sim_method="fft",
        psf_obj=psf_obj,
        gname=gname,
        seed=0,
        ny=npix,
        nx=npix,
        scale=pixel_scale,
        do_shift=False,
        buff=0,
        nrot_per_gal=1,
        mag_zero=mag_zero,
    )[0]
    flux, flux_var = gaussian_flux_variance(
        img_array=gal_array,
        psf_array=psf_array,
        sigma_arcsec=sigma_arcsec,
        sigma_kernel=0.0,
        pixel_scale=pixel_scale,
        noise_variance=noise_std**2.0,
    )
    flux_var2 = anacal.task.gaussian_flux_variance(
        psf_array=psf_array,
        sigma_kernel=0.0,
        sigma_smooth=sigma_arcsec,
        pixel_scale=pixel_scale,
        klim=100
    ) * noise_std**2.0
    np.testing.assert_allclose(flux_var, flux_var2, rtol=0.001, atol=0.01)

    fpfs_config = anacal.fpfs.FpfsConfig(
        sigma_shapelets1=sigma_shapelets,
    )
    fpfs_peaks_dtype = np.dtype([("y", np.float64), ("x", np.float64)])
    det = np.zeros(1, dtype=fpfs_peaks_dtype)
    det["x"] = npix // 2
    det["y"] = npix // 2
    catalog = anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        pixel_scale=pixel_scale,
        mag_zero=mag_zero,
        noise_variance=noise_std**2.0,
        gal_array=gal_array,
        psf_array=psf_array,
        mask_array=None,
        noise_array=None,
        detection=det,
        psf_object=None,
        do_compute_detect_weight=False,
    )
    flux2 = anacal.fpfs.m00_to_flux(
        catalog["fpfs1_m00"],
        pixel_scale=pixel_scale,
        sigma_shapelets=sigma_shapelets
    )[0]
    np.testing.assert_allclose(flux, flux2, rtol=0.001, atol=0.01)


    kwargs = {
        "omega_f": 0.8,
        "v_min": 0.020,
        "omega_v": 0.04,
        "p_min": 0.15,
        "omega_p": 0.05,
    }
    prior = anacal.ngmix.modelPrior()
    det_task = anacal.task.Task(
        scale=pixel_scale,
        sigma_arcsec=sigma_arcsec,
        snr_peak_min=5.0,
        stamp_size=npix,
        image_bound=0,
        num_epochs=0,
        prior=prior,
        force_size=True,
        force_center=True,
        **kwargs,
    )
    blocks = anacal.geometry.get_block_list(
        gal_array.shape[0],
        gal_array.shape[1],
        120,
        120,
        0,
        pixel_scale,
    )
    assert len(blocks) == 1
    catalog2 = det_task.process_image(
        gal_array,
        psf_array,
        variance=noise_std**2.0,
        block_list=blocks,
        a_ini=0.0,
    )
    assert len(catalog2) == 1
    flux3 = catalog2["flux"][0]
    np.testing.assert_allclose(flux, flux3, rtol=0.001, atol=0.01)


    flux4, flux_var4 = gaussian_flux_variance(
        img_array=gal_array,
        psf_array=psf_array,
        sigma_arcsec=sigma_arcsec,
        sigma_kernel=0.2,
        pixel_scale=pixel_scale,
        noise_variance=noise_std**2.0,
    )
    catalog4 = det_task.process_image(
        gal_array,
        psf_array,
        variance=noise_std**2.0,
        block_list=blocks,
        a_ini=0.2,
    )
    np.testing.assert_allclose(catalog4["flux"][0], flux4, rtol=1e-3, atol=0.01)
    catalog5 = det_task.process_image(
        gal_array,
        psf_array,
        variance=noise_std**2.0,
        block_list=blocks,
    )
    np.testing.assert_allclose(catalog5["flux"][0], flux4, rtol=1e-3, atol=0.01)
    return
