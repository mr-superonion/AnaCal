# Pre-rendered test images

The tests do not render images at run time, and AnaCal does not depend on
GalSim, not even for testing. Every image a test needs is stored in this
directory: one multi-extension FITS per test module, with one HDU per image
keyed by `EXTNAME`. The parameters that produced each image are in that HDU's
header, together with the GalSim version (`GSVER`), the generating script
(`GENERAT`), the git hash (`GITHASH`) and the date. `tests/fixtures.py` reads
them: `load("<name>")` returns `{EXTNAME: array}` for `tests/data/<name>.fits`.

Unless stated otherwise, PSF stamps are Moffat profiles drawn on a 0.2″ grid,
shifted by half a pixel so that the PSF centre lands on pixel
(`n // 2`, `n // 2`), AnaCal's PSF convention.

## Files

### Scripts

| File | What it is |
|---|---|
| `make_fixtures.py` | The only renderer. One function per FITS file below (plus three that write the images in `examples/fpfs/`). Run by hand when an image genuinely has to change; never collected by pytest, never installed. |
| `cosmos_sim.py` | The COSMOS galaxy simulator `make_fixtures.py` uses for real-galaxy scenes: `make_isolated_sim` places COSMOS parametric galaxies on a grid with shape-noise-cancelling rotations. It used to be `anacal.simulation`; the same code now also lives in xlens as `xlens.simulator.simplesim`. It needs `src_cosmos.fits` from `$CATSIM_DIR` (see below). |
| `README.md` | This file. |

### `fpfs_*`: FPFS shear estimation (`tests/fpfs/`)

| File | Test | HDUs | Content |
|---|---|---|---|
| `fpfs_process_image.fits` | `test_fpfs_process_image.py`, `test_fpfs_renoise.py`, `test_task_force_consistency.py`, `task/test_renoise_task.py` | `psf`, `gal` | 670×670 COSMOS scene: 10×10 stamps of 64 px plus a 15 px border, 4 rotations per galaxy, `g1 = +0.02`, seed 2, mag zero point 30. PSF: Moffat β 3.5, FWHM 0.6″, truncated at 2.4″. The noisy end-to-end tests share it through `fixtures.cosmos_scene()` / `cosmos_centres()`. |
| `fpfs_shearest.fits` | `test_fpfs_shearest.py` | `psf`, `gal_seed12`, `gal_seed23`, `gal_seed42` | Three 64×768 COSMOS strips of 12 rotations each, `g1 = −0.02`, seeds 12, 23 and 42, mag zero point 27. PSF: Moffat β 3.5, FWHM 0.6″, truncated at 2.4″, `e = (0.02, −0.02)`. |
| `fpfs_force_measure.fits` | `test_fpfs_force_measure.py` | `psf_0`…`psf_3`, `gal_0`…`gal_3` | One elliptical Gaussian (FWHM 0.6″, `e = (0.2, −0.24)`) on 64×64 stamps, in the four (pixel scale, x offset, y offset) cases of `FORCE_CASES`: centred at 0.2″, two offsets of about 2 px at 0.2″, and an offset one at 0.164″; each with its PSF stamp. |
| `fpfs_noise_cov.fits` | `test_fpfs_noise_cov.py` | `psf_moffat35`, `psf_moffat25` | Two PSF stamps for the noise-covariance test: Moffat β 3.5, FWHM 0.6″, truncated, `e = (0.02, −0.02)`; and Moffat β 2.5, FWHM 0.8″. |

### Regression references (not rendered)

Single-HDU files that AnaCal itself produced, read with
`fixtures.load_reference("<name>")`.  They pin the forced FPFS measurement
to its earlier output; re-create them from the current code only when a
change of the output is intended.

| File | Test | Content |
|---|---|---|
| `fpfs_measure_gal.fits`, `fpfs_measure_psf.fits` | `test_fpfs_measure.py` | A 1024×1024 blended scene (float64) and its 64×64 PSF. |
| `fpfs_measure_src.fits` | `test_fpfs_measure.py` | The 12 shapelet modes of the 3274 sources the detector finds on that scene. |
| `fpfs_measure_bfunc_real.fits`, `fpfs_measure_bfunc_imag.fits` | `test_fpfs_measure.py` | The 12 shapelet basis functions on the 64×33 half plane. |
| `fpfs_noise_cov_elem.fits` | `test_fpfs_noise_cov.py` | The 12×12 analytic noise covariance of the `psf_moffat35` PSF. |

### `image*`: image preparation (`tests/test_image.py`, `tests/image/`)

| File | Test | HDUs | Content |
|---|---|---|---|
| `image.fits` | `test_image.py` | `psf_deconv`, `gal_deconv`, `psf_rot`, `psf_rot90` | PSF and one COSMOS galaxy (seed 1, mag zero point 27) for the deconvolution test, and an off-centre elliptical Moffat with its 90° rotation for the rotation test. |
| `image_noise_variance.fits` | `image/test_noise_variance.py` | `psf` | PSF stamp: Moffat β 2.5, FWHM 0.6″, `g = (0.02, −0.02)`. |

### `ngmix_*`: Gaussian model fit, detection weight and background (`tests/ngmix/`)

All use a Moffat β 2.5 PSF with `g = (0.02, −0.02)` (FWHM 0.7″ unless noted) and noise-free galaxies drawn with GalSim.

| File | Test | HDUs | Content |
|---|---|---|---|
| `ngmix_angle.fits` | `test_angle.py` | `psf`, `gal` | Exponential, hlr 0.16″, `e1 = 0.2`, rotated 30°, flux 150. |
| `ngmix_fpfs.fits` | `test_fpfs.py` | `psf`, `gal` | Exponential, hlr 0.21″, `e1 = 0.2`, rotated 30°, flux 150; PSF FWHM 0.8″. |
| `ngmix_gaussfit.fits` | `test_gaussfit.py`, `test_convergence.py` | `psf`, `gal_add`, `gal_g1p_f150`, `gal_g1m_f300`, `gal_g1m_f150`, `gal_g0_a0`, `gal_g0_a90`, `gal_g1p_a45`, `gal_g1p_a90`, `gal_g1p_a135`, `psf_wide`, `gal_wide`, `gal_wide_exp`, `gal_wide_bd` | Exponentials (hlr 0.2″, offset by (−0.2, 0.11) px): a round one for the additive test; an elliptical one (`e = (0.2, −0.1)`) at `g1 = ±0.02`, two fluxes and rotations 0/90°; the `g1 = +0.02` galaxy at 45/90/135°, a ring for the prior-response test. Also a 256×64 strip of four round galaxies sheared by `g1 = 0.02`, as Gaussians, exponentials and bulge+disk, with its PSF. |
| `ngmix_gausscov.fits` | `test_gaussfit.py` | `psf`, `gal` | 256×64 float64 strip of four elliptical Gaussians with known intrinsic covariance (`GAUSSCOV_CASES`), for the covariance-recovery test. |
| `ngmix_wsel.fits` | `test_wsel.py` | `psf`, `gal_init`, `gal_m{0..3}_g{1,2}{m,p}`, `gal_a{0..9}_{0,90}` | Exponential (hlr 0.25″, `e = (0.1, −0.15)`): at mag 26.7–27.0 sheared by ±0.02 in g1 and g2, for the selection-weight response; and at 10 random angles with their 90° rotations. |
| `ngmix_bkg.fits` | `test_bkg.py` | `psf`, `ped_only`, `ped_src`, `bkg_{0..5}_g{1,2}_{m,0,p}`, `wsel_{0..8}_{m,0,p}` | float64 exponentials (`e = (0.1, −0.15)`) at the magnitudes and sizes in `BKG_RESPONSE_CONFIGS` and `WSEL_RESPONSE_CONFIGS`, each at shears −0.002, 0 and +0.002 (`BKG_DG`), for finite-difference checks of the background and selection-weight responses; plus a broad Gaussian pedestal (σ 12″, mag 16) alone and under a mag-22 source. |

### `task_*`: the detection and measurement task (`tests/task/`)

| File | Test | HDUs | Content |
|---|---|---|---|
| `task_detection.fits` | `test_detection.py` | `psf`, `stamp` | 24×24 stamp: exponential, hlr 0.3″, `g1 = 0.03`, mag 23.5. |
| `task_flux_variance.fits` | `test_flux_variance.py` | `psf`, `gal` | One COSMOS galaxy on a 64×64 stamp, mag zero point 30. PSF: Moffat β 3.5, FWHM 0.8″, truncated at 2.4″, `e = (0.02, −0.02)`. See the seed note below. |
| `task_multiband.fits` | `test_multiband.py` | `psf_07`, `psf_09`, `psf_06`, `stamp_07_235`, `stamp_09_240`, `stamp_06_245`, `stamp_07_260` | 32×32 stamps for multi-band detection: three PSFs (FWHM 0.7″, 0.9″, 0.6″) and an exponential (hlr 0.3″, `g1 = 0.03`) at mag 23.5–26.0 under each (key = FWHM, magnitude). |

### `psf_*`: PSF models (`tests/test_psf.py`)

| File | Test | HDUs | Content |
|---|---|---|---|
| `psf_lanczos.fits` | `test_psf.py` | `lanczos_{3,4,5,6,7,9,11}` | GalSim's Lanczos interpolation kernels, the reference for `anacal.psf.lanczos_kernel`. PIFF models are fit with GalSim's interpolant, so the C++ kernel must reproduce GalSim, not the ideal kernel. Each HDU is a 3×1777 table with rows x, raw (`conserve_dc=False`) and dc (`conserve_dc=True`). |

### Images for the examples (written to `examples/fpfs/`)

`make_fixtures.py` also writes `examples/fpfs/isolated_galaxies/isolated_sim.fits`, `examples/fpfs/isolated_galaxies/isolated_highres_sim.fits` (COSMOS scenes for the isolated-galaxy notebooks) and `examples/fpfs/psf_moffat.fits` (the covariance example's PSF).

## Re-rendering

You need GalSim and the COSMOS catalog `src_cosmos.fits`: the GalSim COSMOS
F814W < 25.2 magnitude-limited sample with parametric fits (see
https://galsim-developers.github.io/GalSim/_build/html/real_gal.html). It is not
kept in this repository (29 MB, and it is the only file here that no test
reads). `cosmos_sim.py` looks for it in `$CATSIM_DIR`, the same place
`xlens.simulator.simplesim` reads it from:

    export CATSIM_DIR=/path/to/catsim      # holds src_cosmos.fits
    cd tests/data && python make_fixtures.py

The file used for the stored fixtures is the one that was in this directory
until it was removed. It is still in the git history
(`git show 3749396:tests/data/src_cosmos.fits > $CATSIM_DIR/src_cosmos.fits`).

Re-render only when an image genuinely has to change, because several tests
compare against numbers derived from these exact pixels. With GalSim 2.8.4 and
that catalog, a fresh render is bit-identical to the stored files (checked for
every file above and in `examples/fpfs/`), with one exception:

- **`task_flux_variance.fits`, `gal`.** It is rendered with `seed=0`, and in
  GalSim a seed of 0 means "seed from the operating system". Each render
  therefore picks a different COSMOS galaxy, and the stored image is one such
  draw. The test only needs some galaxy, so it reads the stored pixels and is
  unaffected. If this fixture is ever re-rendered, change the seed to a
  non-zero value so the result is reproducible.

## Checking that nothing needs GalSim

Shadow the package with an empty module on `PYTHONPATH`. This reaches the
notebook kernels nbval starts as well as pytest itself:

    mkdir -p /tmp/nogalsim && echo 'raise ImportError("galsim is not installed")' > /tmp/nogalsim/galsim.py
    PYTHONPATH=/tmp/nogalsim pytest

Nothing may skip.
