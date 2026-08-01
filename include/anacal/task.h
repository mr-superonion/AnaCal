#ifndef ANACAL_TASK_H
#define ANACAL_TASK_H

#include "detector.h"
#include "mask.h"

namespace anacal {
namespace task {


inline double
gaussian_flux_variance(
    const py::array_t<double>& psf_array,
    double sigma_kernel,
    double sigma_smooth,
    double pixel_scale = 1.0,
    double klim = std::numeric_limits<double>::infinity(),
    const std::optional<py::array_t<double>>& noise_corr = std::nullopt
) {
    const ssize_t ny = psf_array.shape(0);
    const ssize_t nx = psf_array.shape(1);
    if (ny <= 0 || nx <= 0) {
        throw std::runtime_error("ngmix Error: PSF image has invalid shape.");
    }
    // enforce even sizes for r2c folding logic below
    if ((ny % 2) != 0 || (nx % 2) != 0) {
        throw std::runtime_error(
            "ngmix Error: this routine assumes even ny and nx."
        );
    }

    // ---- scales: convert to pixels
    const double sigma_pix_rec = sigma_smooth / pixel_scale;  // reconvolution σ (pix)
    const double sigma_pix     = std::sqrt(
        sigma_kernel * sigma_kernel + sigma_smooth * sigma_smooth
    ) / pixel_scale; // meas sigma (pixel)
    if (sigma_pix_rec <= 0.0 || sigma_pix <= 0.0) {
        throw std::runtime_error("ngmix Error: invalid Gaussian widths.");
    }

    // default klim safety (units: same k-units used by Image/filter)
    if (!std::isfinite(klim) || klim <= 0.0) {
        klim = 3.1 / pixel_scale;
    }

    // ---- PSF → Fourier
    Image psf_img(static_cast<int>(nx), static_cast<int>(ny), pixel_scale, true);
    psf_img.set_r(psf_array, true);  // assumes real-space layout compatible with Image
    psf_img.fft();
    const py::array_t<std::complex<double>> psf_fft = psf_img.draw_f();
    const double sigma_k = 1.0 / std::sqrt(
        sigma_kernel * sigma_kernel  + sigma_smooth * sigma_smooth * 2.0
    );
    const Gaussian filter_gauss(sigma_k);

    Image filter_img(static_cast<int>(nx), static_cast<int>(ny), pixel_scale, true);
    filter_img.set_delta_f();             // start from unity impulse in k
    filter_img.filter(filter_gauss);      // multiply by exp(-0.5*k^2/sigma_k^2)
    filter_img.deconvolve(psf_fft, klim); // divide by P(k) within |k|<=klim (and/or floor internally)
    const py::array_t<std::complex<double>> filter_fft = filter_img.draw_f();

    // ---- noise power spectrum
    const ssize_t ky_length = filter_fft.shape(0); // == ny
    const ssize_t kx_length = filter_fft.shape(1); // == nx/2 + 1
    auto filter_fft_r = filter_fft.unchecked<2>();

    py::array_t<double> noise_pow({ky_length, kx_length});
    auto noise_pow_r = noise_pow.mutable_unchecked<2>();
    if (noise_corr.has_value()) {
        if ((*noise_corr).ndim() != 2 ||
            (*noise_corr).shape(0) != ny || (*noise_corr).shape(1) != nx) {
            throw std::runtime_error("ngmix Error: noise correlation image has incompatible shape.");
        }
        Image noise_img(static_cast<int>(nx), static_cast<int>(ny), pixel_scale, true);
        noise_img.set_r(*noise_corr, true);
        noise_img.fft();
        const py::array_t<std::complex<double>> noise_fft = noise_img.draw_f();
        auto noise_fft_r = noise_fft.unchecked<2>();
        for (ssize_t j = 0; j < ky_length; ++j) {
            for (ssize_t i = 0; i < kx_length; ++i) {
                // FFT(C) should be real ≥ 0 up to round-off
                noise_pow_r(j, i) = noise_fft_r(j, i).real();
            }
        }
    } else {
        for (ssize_t j = 0; j < ky_length; ++j) {
            for (ssize_t i = 0; i < kx_length; ++i) {
                noise_pow_r(j, i) = 1.0;  // unit-variance white noise
            }
        }
    }

    // ---- r2c folding along x (even nx): DC once, interior doubled, Nyquist once
    double var_sum = 0.0;

    for (ssize_t j = 0; j < ky_length; ++j) {
        // i = 0 (DC): count once
        {
            const std::complex<double> v = filter_fft_r(j, 0);
            var_sum += std::norm(v) * noise_pow_r(j, 0);
        }

        // interior 1..nx/2-1 : doubled
        for (ssize_t i = 1; i < kx_length - 1; ++i) {
            const std::complex<double> v = filter_fft_r(j, i);
            var_sum += 2.0 * std::norm(v) * noise_pow_r(j, i);
        }

        // i = nx/2 (Nyquist): count once
        {
            const ssize_t iNy = kx_length - 1;     // == nx/2
            const std::complex<double> v = filter_fft_r(j, iNy);
            var_sum += std::norm(v) * noise_pow_r(j, iNy);
        }
    }

    // ---- outer normalization (matches your Python: ff = 4πσ_pix^2, then / (nx*ny))
    const double ff   = 4.0 * M_PI * sigma_pix * sigma_pix;
    const double norm = 1.0 / (static_cast<double>(nx) * static_cast<double>(ny));
    double flux_var   = var_sum * ff * ff * norm;

    if (flux_var < 0.0) flux_var = 0.0;
    return flux_var;
}


class Task {
public:
    // Reference zeropoint at which the base flux-scale thresholds
    // (omega_f/omega_v/fpfs_c0) are defined.  This is the fixed AB
    // nanojansky zeropoint that the measurement normalizes every image onto
    // (xlens MAG_ZERO_AB), so for that path thr_ratio below is exactly 1.0
    // and no rescaling happens.  Callers that feed an image on its native
    // zeropoint (e.g. Euclid VIS MAGZERO = 24.6) still get the correct
    // conversion.
    static constexpr double THRESHOLD_REF_MAG_ZERO = 31.4;
    // stamp dimension
    double scale;
    double sigma_arcsec;
    double snr_peak_min, omega_f, omega_v;
    const ngmix::modelPrior prior;
    int stamp_size, ss2;
    int image_bound;
    int num_epochs;
    bool force_size, force_center;
    ngmix::GaussFit fitter;
    double sigma_arcsec_det;

    Task(
        double scale,
        double sigma_arcsec,
        double snr_peak_min,
        double omega_f,
        double omega_v,
        const std::optional<ngmix::modelPrior>& prior=std::nullopt,
        int stamp_size=64,
        int image_bound=0,
        int num_epochs=3,
        bool force_size=false,
        bool force_center=false,
        double fpfs_c0=1.0,
        double mag_zero=THRESHOLD_REF_MAG_ZERO
    ) : scale(scale), sigma_arcsec(sigma_arcsec), snr_peak_min(snr_peak_min),
        omega_f(omega_f), omega_v(omega_v),
        prior(prior ? *prior : ngmix::modelPrior()),
        stamp_size(stamp_size), image_bound(image_bound),
        num_epochs(num_epochs), fitter(
            scale, sigma_arcsec, stamp_size,
            force_size, force_center,
            fpfs_c0 * std::pow(
                10.0, (mag_zero - THRESHOLD_REF_MAG_ZERO) / 2.5
            )
        )
    {
        if (stamp_size % 2 != 0 ) {
            throw std::invalid_argument("nx or ny is not even number");
        }
        if (sigma_arcsec <= 0) {
            throw std::invalid_argument("sigma_arcsec must be positive");
        }
        this->sigma_arcsec_det = detection_sigma(sigma_arcsec);

        // Own the mag_zero-dependent THRESHOLD scaling here (previously duplicated
        // in every caller). omega_f/omega_v/fpfs_c0 are BASE thresholds
        // defined at THRESHOLD_REF_MAG_ZERO; scale them to the image's ``mag_zero``.
        // The image itself is normalized upstream (anacal.fpfs.rescale_image_to_
        // zeropoint), so no image rescale happens here.
        const double thr_ratio =
            std::pow(10.0, (mag_zero - THRESHOLD_REF_MAG_ZERO) / 2.5);
        this->omega_f *= thr_ratio;
        this->omega_v *= thr_ratio;
    };

    inline void
    prepare_indices(
        std::vector<table::galNumber>& catalog,
        geometry::block & block
    ) {
        double x_min = block.xmin * block.scale;
        double y_min = block.ymin * block.scale;
        double x_max = block.xmax * block.scale;
        double y_max = block.ymax * block.scale;
        std::size_t nrow = catalog.size();
        std::vector<std::size_t> indices;
        indices.reserve(static_cast<std::size_t>(nrow / 4));
        for (std::size_t i = 0; i < nrow; ++i) {
            const table::galNumber & src = catalog[i];
            if ((src.x1_det >= x_min) &&
                (src.x1_det < x_max) &&
                (src.x2_det >= y_min) &&
                (src.x2_det < y_max)
            ) {
                indices.push_back(i);
            }
        }
        block.indices = indices;
        return;
    };

    // Which PSF to use for a block: the block's own stamp when it has one of
    // the right rank and size, otherwise the image-wide one.  Written once
    // here because the same choice is made at three points in process_image.
    // For a 2-D PSF this is the condition that was in place before multi-band
    // support, so single-band behaviour is unchanged.
    inline const py::array_t<double>&
    choose_psf(
        const geometry::block & block,
        const py::array_t<double>& psf_array
    ) const {
        const ssize_t nd = psf_array.ndim();
        const bool shape_ok = (
            (psf_array.shape(nd - 2) == this->stamp_size) &&
            (psf_array.shape(nd - 1) == this->stamp_size)
        );
        if ((block.psf_array.ndim() == nd) && shape_ok) {
            return block.psf_array;
        }
        return psf_array;
    };

    inline std::vector<table::galNumber>
    detect_block(
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        const std::vector<double>& variance,
        const geometry::block & block,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
        const std::optional<std::vector<double>>& weights=std::nullopt
    ) {
        std::vector<table::galNumber> catalog = detector::find_peaks(
            img_array,
            psf_array,
            this->sigma_arcsec,
            this->snr_peak_min,
            variance,
            this->omega_f,
            this->omega_v,
            block,
            noise_array,
            this->image_bound,
            weights
        );
        for (table::galNumber& src : catalog) {
            src = src.decentralize(block);
        }
        return catalog;
    };

    inline void
    measure_block(
        std::vector<table::galNumber>& catalog,
        std::vector<table::galNumber>& catalog_model,
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        const std::vector<double>& variance,
        const geometry::block & block,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
        int run_id=0,
        const std::optional<std::vector<double>>& weights=std::nullopt
    ) {
        if (block.indices.empty()) return;
        for (std::size_t idx : block.indices) {
            catalog[idx] = catalog[idx].centralize(block);
            catalog_model[idx] = catalog_model[idx].centralize(block);
        }
        this->fitter.process_block_impl(
            catalog,
            catalog_model,
            img_array,
            psf_array,
            this->prior,
            this->num_epochs,
            variance,
            block,
            noise_array,
            run_id,
            weights
        );
        for (std::size_t idx : block.indices) {
            catalog[idx] = catalog[idx].decentralize(block);
            catalog_model[idx] = catalog_model[idx].decentralize(block);
        }
        return;
    };

    inline py::array_t<table::galRow>
    process_image(
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        const varianceArg& variance,
        std::vector<geometry::block>& block_list,
        const std::optional<py::array_t<table::galRow>>& detection=std::nullopt,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
        const std::optional<py::array_t<int16_t>>& mask_array=std::nullopt,
        double a_ini=0.2,
        bool do_measure=true,
        bool do_fpfs=true
    ) {
        // ``img_array`` may be a plain (ny, nx) image or an
        // (nband, ny, nx) stack; ``psf_array`` and ``noise_array`` follow it,
        // and ``variance`` carries one value per band.  Several bands are
        // combined into one qimage after each band's PSF has been removed --
        // see prepare_data_block_coadd in image.h.
        const std::vector<double> variance_in = to_variance_vector(variance);
        check_band_stack(img_array, psf_array, variance_in, noise_array);

        std::vector<double> variance_use = variance_in;
        if (noise_array.has_value()) {
            for (double& v : variance_use) {
                v = v * 2.0;
            }
        }

        // Band weights depend on the PSF, which varies from block to block, so
        // they are worked out per block and handed to BOTH the detection and
        // the measurement.  Passing them explicitly is what guarantees the two
        // stages see the same coadd.
        auto block_weights = [&](const py::array_t<double>& psf,
                                 const geometry::block & block) {
            return band_weights(
                block.scale, this->sigma_arcsec_det, psf, variance_use
            );
        };

        std::vector<table::galNumber> catalog;
        if (detection.has_value()) {
            catalog = table::array_to_objlist(
                *detection
            );
        } else {
            for (const geometry::block & block: block_list) {
                const py::array_t<double>& psf = choose_psf(block, psf_array);
                std::vector<table::galNumber> det = detect_block(
                    img_array,
                    psf,
                    variance_use,
                    block,
                    noise_array,
                    block_weights(psf, block)
                );
                catalog.reserve(catalog.size() + det.size());
                for (const table::galNumber& det_src : det) {
                    table::galNumber src = det_src;
                    src.model.a1 = math::qnumber(a_ini);
                    src.model.a2 = math::qnumber(a_ini);
                    catalog.push_back(src);
                }
            }
        }

        this->fitter.do_fpfs = do_fpfs;

        if (do_measure) {
            // Flux error of the coadd: each band contributes its own Gaussian
            // flux variance, combined with the same w_b^2 weighting used for
            // the image itself.  One band reduces to the previous expression.
            auto compute_flux_errors = [&](const py::array_t<double>& psf,
                                           const std::vector<double>& w) {
                double var0 = 0.0;
                double var2 = 0.0;
                for (std::size_t b = 0; b < w.size(); ++b) {
                    const py::array_t<double> psf_b = band_view(
                        psf, static_cast<ssize_t>(b)
                    );
                    const double ww = w[b] * w[b] * variance_use[b];
                    var0 = var0 + ww * std::max(0.0, gaussian_flux_variance(
                        psf_b, 0.0, this->sigma_arcsec, this->scale
                    ));
                    var2 = var2 + ww * std::max(0.0, gaussian_flux_variance(
                        psf_b, 0.2, this->sigma_arcsec, this->scale
                    ));
                }
                std::array<double, 2> errs{};
                errs[0] = std::sqrt(var0);
                errs[1] = std::sqrt(var2);
                return errs;
            };

            for (geometry::block & block: block_list) {
                prepare_indices(
                    catalog,
                    block
                );

                if (block.indices.empty()) {
                    continue;
                }

                const py::array_t<double>& psf = choose_psf(block, psf_array);
                const std::array<double, 2> block_flux_errs = compute_flux_errors(
                    psf, block_weights(psf, block)
                );
                for (std::size_t idx : block.indices) {
                    catalog[idx].flux_gauss0_err = block_flux_errs[0];
                    catalog[idx].flux_gauss2_err = block_flux_errs[1];
                }
            }

            std::vector<table::galNumber> catalog_model = catalog;
            for (table::galNumber & src : catalog_model) {
                src.model.F = src.model.F * src.wdet;
            }
            for (const geometry::block & block: block_list) {
                if (block.indices.empty()) {
                    continue;
                }
                const py::array_t<double>& psf = choose_psf(block, psf_array);
                measure_block(
                    catalog,
                    catalog_model,
                    img_array,
                    psf,
                    variance_use,
                    block,
                    noise_array,
                    0, // run_id
                    block_weights(psf, block)
                );
            }

            if (mask_array.has_value()) {
                mask::add_pixel_mask_column(
                    catalog,
                    *mask_array,
                    this->sigma_arcsec_det * 1.5,
                    scale
                );
            }
        }
        return table::objlist_to_array(catalog);
    };
};

void pyExportTask(py::module_& m);

} // task
} // anacal

#endif // ANACAL_TASK_H
