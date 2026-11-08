"""The Task's exported ``flux`` is the PROFILED flux of the forced Gaussian
model on the deconvolved, re-smoothed image: for a Gaussian template this
is the matched-aperture flux of the same width, so it equals the Fourier
aperture flux of sigma^2 + a_ini^2 (and the forced FPFS m00 flux)."""

import anacal
import numpy as np

from ..fixtures import load


def gaussian_aperture_flux(
    img_array,
    psf_array,
    sigma_arcsec,
    sigma_kernel,
    pixel_scale=1.0,
    eps=1e-6,
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
    img_f = np.fft.fft2(np.fft.ifftshift(img_array))
    return np.sum(np.conj(W*H) * img_f).real * ff / (nx * ny)


def test_profiled_flux():
    mag_zero = 30
    noise_std = 0.3
    pixel_scale = 0.2
    sigma_arcsec = 0.38
    sigma_shapelets = sigma_arcsec * np.sqrt(2.0)
    npix = 64
    # pre-rendered (tests/data/task_flux_variance.fits): a sheared,
    # truncated Moffat PSF (beta 3.5, fwhm 0.8) and one COSMOS galaxy
    # (seed 0, "g1-0", mag_zero 30) at 0.2 arcsec/pixel
    fix = load("task_flux_variance")
    psf_array = fix["psf"]
    psf_array = np.asarray(
        anacal.psf.resize_array(
            psf_array, (npix, npix)
        ),
        dtype=np.float64,
    )
    gal_array = fix["gal"]
    flux = gaussian_aperture_flux(
        img_array=gal_array,
        psf_array=psf_array,
        sigma_arcsec=sigma_arcsec,
        sigma_kernel=0.0,
        pixel_scale=pixel_scale,
    )

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
    )
    flux2 = anacal.fpfs.m00_to_flux(
        catalog["fpfs1_m00"],
        sigma_shapelets=sigma_shapelets,
    )[0]
    np.testing.assert_allclose(flux, flux2, rtol=0.001, atol=0.01)

    kwargs = {
        "omega_f": 0.8,
        "omega_v": 0.04,
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
    # cell_overlap must be at least twice the background kernel reach
    # (2 * (3 arcsec / scale + 1) = 32 pixels here) so the local background is
    # never estimated from pixels outside the cell.  120 - 32 = 88 still
    # exceeds the 64-pixel image, so this remains a single cell.
    cells = anacal.geometry.get_cell_list(
        img_nx=gal_array.shape[1],
        img_ny=gal_array.shape[0],
        cell_nx=120,
        cell_ny=120,
        cell_overlap=32,
        scale=pixel_scale,
    )
    assert len(cells) == 1
    flux4 = gaussian_aperture_flux(
        img_array=gal_array,
        psf_array=psf_array,
        sigma_arcsec=sigma_arcsec,
        sigma_kernel=0.2,
        pixel_scale=pixel_scale,
    )
    # At num_epochs = 0 the exported flux is the PROFILED model flux of
    # the forced Gaussian (covariance a_ini^2 I, centre at the detection),
    # F* = sum(d m~) / sum(m~^2) over the fit window of the deconvolved,
    # re-smoothed image: forced photometry with a Gaussian profile.  For
    # a Gaussian template this is the matched-aperture flux of the same
    # width, so it equals the sigma^2 + a_ini^2 aperture flux to 1e-6.

    def profiled_flux(cat, a_ini):
        q = anacal.image.ImageQ(
            nx=npix, ny=npix, scale=pixel_scale, sigma_arcsec=sigma_arcsec,
            klim=100.0,
        )
        data = q.prepare_qnumber_image(
            gal_array, psf_array, xcen=npix // 2, ycen=npix // 2
        )[0]
        m = anacal.ngmix.NgmixGaussian()
        m.set_axes(
            anacal.math.qnumber(a_ini), anacal.math.qnumber(a_ini),
            anacal.math.qnumber(0.0),
        )
        m.x1 = anacal.math.qnumber(cat["x1"][0])
        m.x2 = anacal.math.qnumber(cat["x2"][0])
        m.F = anacal.math.qnumber(1.0)
        unit = m.get_image_stamp(npix, npix, pixel_scale, sigma_arcsec)[0]
        yy, xx = np.mgrid[0:npix, 0:npix]
        win = (
            (xx * pixel_scale - m.x1.v) ** 2 + (yy * pixel_scale - m.x2.v) ** 2
        ) < 3.5 ** 2
        return np.sum(data[win] * unit[win]) / np.sum(unit[win] ** 2)

    catalog2 = det_task.process_image(
        gal_array,
        psf_array,
        variance=noise_std**2.0,
        cell_list=cells,
        a_ini=0.0,
    )
    assert len(catalog2) == 1
    np.testing.assert_allclose(
        catalog2["flux"][0], profiled_flux(catalog2, 0.0), rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(catalog2["flux"][0], flux, rtol=1e-6, atol=0)
    catalog4 = det_task.process_image(
        gal_array,
        psf_array,
        variance=noise_std**2.0,
        cell_list=cells,
        a_ini=0.2,
    )
    np.testing.assert_allclose(
        catalog4["flux"][0], profiled_flux(catalog4, 0.2), rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(catalog4["flux"][0], flux4, rtol=1e-6, atol=0)
    # a_ini defaults to 0.2
    catalog5 = det_task.process_image(
        gal_array,
        psf_array,
        variance=noise_std**2.0,
        cell_list=cells,
    )
    np.testing.assert_allclose(catalog5["flux"][0], catalog4["flux"][0])
