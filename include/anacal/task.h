#ifndef ANACAL_TASK_H
#define ANACAL_TASK_H

#include "detector.h"
#include "mask.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>

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
    // ---- input checks
    if (psf_array.ndim() != 2) {
        throw std::runtime_error("ngmix Error: PSF image must be 2-dimensional.");
    }
    if (sigma_smooth <= 0.0) {
        throw std::runtime_error("ngmix Error: sigma_smooth must be positive.");
    }
    if (sigma_kernel < 0.0) {
        throw std::runtime_error("ngmix Error: sigma_kernel must be non-negative.");
    }
    if (pixel_scale <= 0.0) {
        throw std::runtime_error("ngmix Error: pixel_scale must be positive.");
    }

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
    // stamp dimension
    double scale;
    double sigma_arcsec;
    double snr_peak_min, omega_f, v_min, omega_v, p_min, omega_p;
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
        double v_min,
        double omega_v,
        double p_min,
        double omega_p,
        const std::optional<ngmix::modelPrior>& prior=std::nullopt,
        int stamp_size=64,
        int image_bound=0,
        int num_epochs=3,
        bool force_size=false,
        bool force_center=false,
        double fpfs_c0=1.0
    ) : scale(scale), sigma_arcsec(sigma_arcsec), snr_peak_min(snr_peak_min),
        omega_f(omega_f), v_min(v_min), omega_v(omega_v),
        p_min(p_min), omega_p(omega_p),
        prior(prior ? *prior : ngmix::modelPrior()),
        stamp_size(stamp_size), image_bound(image_bound),
        num_epochs(num_epochs), fitter(
            scale, sigma_arcsec, stamp_size,
            force_size, force_center,
            fpfs_c0
        )
    {
        if (stamp_size % 2 != 0 ) {
            throw std::invalid_argument("nx or ny is not even number");
        }
        if (sigma_arcsec <= 0) {
            throw std::invalid_argument("sigma_arcsec must be positive");
        }
        this->sigma_arcsec_det = sigma_arcsec * 1.414;
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

    inline std::vector<table::galNumber>
    detect_block(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        double variance,
        const geometry::block & block,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {
        std::vector<table::galNumber> catalog = detector::find_peaks(
            img_array,
            psf_array,
            this->sigma_arcsec,
            this->snr_peak_min,
            variance,
            this->omega_f,
            this->v_min,
            this->omega_v,
            this->p_min,
            this->omega_p,
            block,
            noise_array,
            this->image_bound
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
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        double variance,
        const geometry::block & block,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        int run_id=0
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
            run_id
        );
        for (std::size_t idx : block.indices) {
            catalog[idx] = catalog[idx].decentralize(block);
            catalog_model[idx] = catalog_model[idx].decentralize(block);
        }
        return;
    };

    inline py::array_t<table::galRow>
    process_image(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        double variance,
        std::vector<geometry::block>& block_list,
        const std::optional<py::array_t<table::galRow>>& detection=std::nullopt,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        const std::optional<py::array_t<int16_t>>& mask_array=std::nullopt,
        double a_ini=0.2
    ) {

        double variance_use;
        if (noise_array.has_value()) {
            variance_use = variance * 2.0;
        } else {
            variance_use = variance;
        }

        const double sigma_smooth = this->sigma_arcsec;
        auto compute_flux_errors = [&](const py::array_t<double>& psf) {
            const double flux_gauss0_var = gaussian_flux_variance(
                psf,
                0.0,
                sigma_smooth,
                this->scale
            );
            const double flux_gauss2_var = gaussian_flux_variance(
                psf,
                0.2,
                sigma_smooth,
                this->scale
            );
            const double flux_gauss4_var = gaussian_flux_variance(
                psf,
                0.4,
                sigma_smooth,
                this->scale
            );
            std::array<double, 3> errs{};
            errs[0] = std::sqrt(std::max(0.0, flux_gauss0_var) * variance_use);
            errs[1] = std::sqrt(std::max(0.0, flux_gauss2_var) * variance_use);
            errs[2] = std::sqrt(std::max(0.0, flux_gauss4_var) * variance_use);
            return errs;
        };

        const std::array<double, 3> default_flux_errs = compute_flux_errors(psf_array);

        std::vector<table::galNumber> catalog;
        if (detection.has_value()) {
            catalog = table::array_to_objlist(
                *detection
            );
        } else {
            for (const geometry::block & block: block_list) {
                py::array_t<double> psf;
                if (
                    block.psf_array.ndim() == 2 &&
                    psf_array.shape(0) == this->stamp_size &&
                    psf_array.shape(1) == this->stamp_size
                ) {
                    psf = block.psf_array;
                } else {
                    psf = psf_array;
                }
                std::vector<table::galNumber> det = detect_block(
                    img_array,
                    psf,
                    variance_use,
                    block,
                    noise_array
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

        for (table::galNumber & src : catalog) {
            src.flux_gauss0_err = default_flux_errs[0];
            src.flux_gauss2_err = default_flux_errs[1];
            src.flux_gauss4_err = default_flux_errs[2];
        }

        for (geometry::block & block: block_list) {
            prepare_indices(
                catalog,
                block
            );

            if (block.indices.empty()) {
                continue;
            }

            const bool has_custom_psf = (
                block.psf_array.ndim() == 2 &&
                psf_array.shape(0) == this->stamp_size &&
                psf_array.shape(1) == this->stamp_size
            );
            if (!has_custom_psf) {
                continue;
            }

            const std::array<double, 3> block_flux_errs = compute_flux_errors(
                block.psf_array
            );
            for (std::size_t idx : block.indices) {
                catalog[idx].flux_gauss0_err = block_flux_errs[0];
                catalog[idx].flux_gauss2_err = block_flux_errs[1];
                catalog[idx].flux_gauss4_err = block_flux_errs[2];
            }
        }

        std::vector<table::galNumber> catalog_model = catalog;
        for (table::galNumber & src : catalog_model) {
            src.model.F = src.model.F * src.wdet;
        }
        for (const geometry::block & block: block_list) {
            py::array_t<double> psf;
            if (
                block.psf_array.ndim() == 2 &&
                psf_array.shape(0) == this->stamp_size &&
                psf_array.shape(1) == this->stamp_size
            ) {
                psf = block.psf_array;
            } else {
                psf = psf_array;
            }
            measure_block(
                catalog,
                catalog_model,
                img_array,
                psf,
                variance_use,
                block,
                noise_array,
                0 // run_id
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
        return table::objlist_to_array(catalog);
    };
};

void pyExportTask(py::module_& m);

} // task
} // anacal

#endif // ANACAL_TASK_H
