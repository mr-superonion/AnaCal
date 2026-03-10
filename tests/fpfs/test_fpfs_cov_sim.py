import anacal
import galsim
import numpy as np
import numpy.lib.recfunctions as rfn


def test_covariance_from_simulation():
    """Validate prepare_covariance by comparing with empirical covariance
    from many noise realizations."""
    pixel_scale = 0.2
    noise_std = 0.6
    noise_variance = noise_std**2.0
    ngrid = 64
    sigma_shapelets = 0.5374011
    nreal = 20000
    seed = 42

    # --- PSF ---
    psf_obj = galsim.Moffat(beta=2.5, fwhm=0.8)
    psf_array = (
        psf_obj.shift(0.5 * pixel_scale, 0.5 * pixel_scale)
        .drawImage(nx=ngrid, ny=ngrid, scale=pixel_scale)
        .array
    )

    # --- FpfsTask ---
    ftask = anacal.fpfs.FpfsTask(
        npix=ngrid,
        psf_array=psf_array,
        pixel_scale=pixel_scale,
        sigma_shapelets=sigma_shapelets,
        do_detection=True,
        noise_variance=noise_variance,
    )

    # --- Analytical covariance ---
    cov_analytic = ftask.prepare_covariance(variance=noise_variance)
    ncol = cov_analytic.shape[0]

    # --- Force measurement coordinates (center of stamp) ---
    coords = np.array(
        [(ngrid / 2.0, ngrid / 2.0)],
        dtype=[("y", "f8"), ("x", "f8")],
    )

    # --- Run many noise realizations ---
    # prepare_covariance doubles noise variance because the full pipeline
    # adds a noise bias measurement (from noise_array) to the galaxy
    # To match, we provide two independent noise images per realization.
    rng = np.random.RandomState(seed)
    results = np.empty((nreal, ncol))
    for i in range(nreal):
        noise_image = rng.normal(scale=noise_std, size=(ngrid, ngrid)).astype(
            np.float64
        )
        noise_image2 = rng.normal(scale=noise_std, size=(ngrid, ngrid)).astype(
            np.float64
        )
        src = ftask.run(
            gal_array=noise_image,
            psf=psf_array,
            det=coords,
            noise_array=noise_image2,
        )
        row = rfn.structured_to_unstructured(src["data"])
        results[i] = row[0]

    # --- Empirical covariance ---
    cov_empirical = np.cov(results, rowvar=False)

    # The empirical covariance should agree with the analytical one.
    # With 20000 realizations, we expect ~1% relative accuracy on
    # individual elements. Use a generous tolerance for robustness.
    np.testing.assert_allclose(
        cov_empirical,
        cov_analytic,
        atol=3.0 * np.max(np.abs(cov_analytic)) / np.sqrt(nreal),
        rtol=0.05,
    )

    # Also check that std_modes matches the diagonal
    std_empirical = np.std(results, axis=0)
    np.testing.assert_allclose(
        std_empirical,
        ftask.std_modes,
        rtol=0.05,
    )
