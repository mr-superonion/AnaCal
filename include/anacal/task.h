#ifndef ANACAL_TASK_H
#define ANACAL_TASK_H

#include "detector.h"
#include "mask.h"

namespace anacal {
namespace task {

// Band-sliced white-noise variant, free of numpy allocations so it can
// run with the GIL released (the PSF stack is only read through unchecked
// accessors).  The full function below keeps the optional noise
// correlation for the Python binding.
inline double
gaussian_flux_variance_band(
    const py::array_t<double>& psf_stack,
    ssize_t band,
    double sigma_kernel,
    double sigma_smooth,
    double pixel_scale = 1.0,
    double klim = std::numeric_limits<double>::infinity()
) {
    const ssize_t nd = psf_stack.ndim();
    const ssize_t ny = psf_stack.shape(nd - 2);
    const ssize_t nx = psf_stack.shape(nd - 1);
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
    const double sigma_pix_rec = sigma_smooth / pixel_scale;
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

    // ---- PSF -> Fourier
    ImageLease psf_lease(
        static_cast<int>(nx), static_cast<int>(ny), pixel_scale, true
    );
    Image& psf_img = psf_lease.get();
    psf_img.set_r_band(psf_stack, band, true);
    psf_img.fft();
    const std::vector<std::complex<double>> psf_fft = psf_img.draw_f_vec();
    const double sigma_k = 1.0 / std::sqrt(
        sigma_kernel * sigma_kernel  + sigma_smooth * sigma_smooth * 2.0
    );
    const Gaussian filter_gauss(sigma_k);

    ImageLease filter_lease(
        static_cast<int>(nx), static_cast<int>(ny), pixel_scale, true
    );
    Image& filter_img = filter_lease.get();
    filter_img.set_delta_f();
    filter_img.filter(filter_gauss);
    filter_img.deconvolve(psf_fft, klim);
    const complex2* filter_fft = filter_img.view_f();

    // ---- r2c folding along x (even nx) with unit-variance white noise:
    // DC once, interior doubled, Nyquist once
    const ssize_t ky_length = ny;
    const ssize_t kx_length = nx / 2 + 1;
    double var_sum = 0.0;

    for (ssize_t j = 0; j < ky_length; ++j) {
        // i = 0 (DC): count once
        {
            const ssize_t index = j * kx_length;
            const std::complex<double> v(
                filter_fft[index][0], filter_fft[index][1]
            );
            var_sum += std::norm(v) * 1.0;
        }

        // interior 1..nx/2-1 : doubled
        for (ssize_t i = 1; i < kx_length - 1; ++i) {
            const ssize_t index = j * kx_length + i;
            const std::complex<double> v(
                filter_fft[index][0], filter_fft[index][1]
            );
            var_sum += 2.0 * std::norm(v) * 1.0;
        }

        // i = nx/2 (Nyquist): count once
        {
            const ssize_t index = j * kx_length + (kx_length - 1);
            const std::complex<double> v(
                filter_fft[index][0], filter_fft[index][1]
            );
            var_sum += std::norm(v) * 1.0;
        }
    }

    const double ff   = 4.0 * M_PI * sigma_pix * sigma_pix;
    const double norm = 1.0 / (static_cast<double>(nx) * static_cast<double>(ny));
    double flux_var   = var_sum * ff * ff * norm;

    if (flux_var < 0.0) flux_var = 0.0;
    return flux_var;
}


inline double
gaussian_flux_variance(
    const py::array_t<double>& psf_array,
    double sigma_kernel,
    double sigma_smooth,
    double pixel_scale = 1.0,
    double klim = std::numeric_limits<double>::infinity(),
    const std::optional<py::array_t<double>>& noise_corr = std::nullopt
) {
    // White noise IS the band-sliced implementation's case (its 1.0
    // noise power equals the unit spectrum built below); only the
    // noise-correlation path needs the numpy machinery here.
    if (!noise_corr.has_value()) {
        return gaussian_flux_variance_band(
            psf_array, 0, sigma_kernel, sigma_smooth, pixel_scale, klim
        );
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


// Recompute every source's cell_id from its position.  The cells'
// INNER regions tile the image without overlap, so each position has
// exactly one owner; a source outside all inner regions (padding or the
// image_bound margin -- possible for forced positions) is given the
// nearest cell.  This is what makes the ``cell_id == cell.index``
// guard in the measurement stage trustworthy for ANY input catalog:
// cell_id is not an input column, and a stale or foreign value can no
// longer leave a source silently unmeasured.
inline void
assign_cell_ids(
    std::vector<table::galNumber>& catalog,
    const std::vector<geometry::cell>& cell_list
) {
    if (cell_list.empty()) return;

    // The cells' inner regions tile the image on a REGULAR grid, so
    // ownership is a row/column binary search instead of a scan over
    // every cell (which was O(sources x cells)).  The half-open
    // [x0, x1) rule is unchanged: a source on a shared inner edge
    // belongs to the cell on its right/top, matching the detection
    // scan bounds.
    std::vector<double> xs, ys;
    xs.reserve(cell_list.size());
    ys.reserve(cell_list.size());
    for (const geometry::cell & b : cell_list) {
        xs.push_back(b.xmin_in * b.scale);
        ys.push_back(b.ymin_in * b.scale);
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::sort(ys.begin(), ys.end());
    ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    const std::size_t ncol = xs.size();
    auto col_of = [&](double v) {
        // index of the last column start <= v; only valid when
        // v >= xs.front()
        return static_cast<std::size_t>(
            std::upper_bound(xs.begin(), xs.end(), v) - xs.begin()
        ) - 1;
    };
    auto row_of = [&](double v) {
        return static_cast<std::size_t>(
            std::upper_bound(ys.begin(), ys.end(), v) - ys.begin()
        ) - 1;
    };
    std::vector<int> grid(ncol * ys.size(), -1);
    for (std::size_t ib = 0; ib < cell_list.size(); ++ib) {
        const geometry::cell & b = cell_list[ib];
        grid[
            row_of(b.ymin_in * b.scale) * ncol
            + col_of(b.xmin_in * b.scale)
        ] = static_cast<int>(ib);
    }

    // Original nearest-cell scan, kept for sources OUTSIDE every
    // inner region (padding or the image_bound margin -- possible for
    // forced positions) and for cell lists that are not a full grid
    // (e.g. cells dropped for missing PSFs upstream).
    auto nearest_scan = [&](const table::galNumber & src) {
        int best = cell_list.front().index;
        double best_d2 = std::numeric_limits<double>::infinity();
        for (const geometry::cell & b : cell_list) {
            double x0 = b.xmin_in * b.scale;
            double x1 = b.xmax_in * b.scale;
            double y0 = b.ymin_in * b.scale;
            double y1 = b.ymax_in * b.scale;
            if ((src.x1_det >= x0) && (src.x1_det < x1) &&
                (src.x2_det >= y0) && (src.x2_det < y1)) {
                return b.index;
            }
            double dx = std::max({x0 - src.x1_det, 0.0, src.x1_det - x1});
            double dy = std::max({y0 - src.x2_det, 0.0, src.x2_det - y1});
            double d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                best_d2 = d2;
                best = b.index;
            }
        }
        return best;
    };

    for (table::galNumber & src : catalog) {
        int best = -1;
        if ((src.x1_det >= xs.front()) && (src.x2_det >= ys.front())) {
            const int ib = grid[
                row_of(src.x2_det) * ncol + col_of(src.x1_det)
            ];
            if (ib >= 0) {
                const geometry::cell & b = cell_list[ib];
                if ((src.x1_det < b.xmax_in * b.scale) &&
                    (src.x2_det < b.ymax_in * b.scale)) {
                    best = b.index;
                }
            }
        }
        if (best < 0) {
            best = nearest_scan(src);
        }
        src.cell_id = best;
    }
    return;
};


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

    // Which PSF to use for a cell: the cell's own stamp when it has one of
    // the right rank and size, otherwise the exposure-wide one.  Written once
    // here because the same choice is made at three points in process_image.
    // For a 2-D PSF this is the condition that was in place before multi-band
    // support, so single-band behaviour is unchanged.
    inline const py::array_t<double>&
    choose_psf(
        const geometry::cell & cell,
        const py::array_t<double>& psf_array
    ) const {
        const ssize_t nd = psf_array.ndim();
        const bool shape_ok = (
            (psf_array.shape(nd - 2) == this->stamp_size) &&
            (psf_array.shape(nd - 1) == this->stamp_size)
        );
        if ((cell.psf_array.ndim() == nd) && shape_ok) {
            return cell.psf_array;
        }
        return psf_array;
    };

    inline std::vector<table::galNumber>
    detect_cell(
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        const std::vector<double>& variance,
        const geometry::cell & cell,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
        const std::optional<std::vector<double>>& weights=std::nullopt,
        const std::optional<double>& multiband_coadd_variance=std::nullopt
    ) {
        // The impl variant: process_image validated the stacks already
        // and holds the GIL release, so the public wrapper's re-check
        // and nested release are skipped.
        std::vector<table::galNumber> catalog = detector::find_peaks_impl(
            img_array,
            psf_array,
            this->sigma_arcsec,
            this->snr_peak_min,
            variance,
            this->omega_f,
            this->omega_v,
            cell,
            noise_array,
            this->image_bound,
            weights,
            multiband_coadd_variance
        );
        for (table::galNumber& src : catalog) {
            src.decentralize(cell);
        }
        return catalog;
    };

    // Measure the sources a cell OWNS.  ``rows`` holds exactly those
    // sources (detected in, or assigned to, this cell's inner region) and
    // every one of them is measured.  Overlap neighbours are no longer
    // carried along: they were only needed to render neighbour models for
    // deblending, which was removed (see deblend_plan.txt).
    inline void
    measure_cell(
        std::vector<table::galNumber>& rows,
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        const std::vector<double>& variance,
        const geometry::cell & cell,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
        const std::optional<std::vector<double>>& weights=std::nullopt,
        const std::optional<double>& variance_meas=std::nullopt,
        const std::optional<double>& variance_det=std::nullopt,
        const std::optional<int>& mask_value_max=std::nullopt
    ) {
        if (rows.empty()) return;
        for (table::galNumber & src : rows) {
            src.centralize(cell);
        }
        this->fitter.process_cell_impl(
            rows,
            img_array,
            psf_array,
            this->prior,
            this->num_epochs,
            variance,
            cell,
            noise_array,
            weights,
            variance_meas,
            variance_det,
            mask_value_max
        );
        for (table::galNumber & src : rows) {
            src.decentralize(cell);
        }
        return;
    };

    inline py::array_t<table::galRow>
    process_image(
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        const varianceArg& variance,
        const std::vector<geometry::cell>& cell_list,
        const std::optional<py::array_t<table::galRow>>& detection=std::nullopt,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
        const std::optional<py::array_t<int16_t>>& mask_array=std::nullopt,
        double a_ini=0.2,
        bool do_measure=true,
        bool do_fpfs=true,
        // Sources whose mask_value exceeds this are SKIPPED by the
        // measurement (rows kept with default values); internal
        // detections are stamped from mask_array before the check,
        // external catalogs carry their own mask_value column.
        const std::optional<int>& mask_value_max=std::nullopt
    ) {
        // From here to the reset() before the output conversion,
        // everything works on C++ containers and READS the input arrays
        // through unchecked accessors only -- the prologue below (variance
        // normalization, band-stack validation, weights lambda, detection
        // conversion) is pure C++ plus GIL-free ndim/shape struct reads,
        // and validation errors unwind safely through the release -- so the
        // GIL is dropped for the whole call; this is what lets callers
        // thread over cells.  (The FFT engine is lock-free: see
        // image.h.)
        std::optional<ScopedGilRelease> release_for_cells;
        release_for_cells.emplace();

        // ``img_array`` may be a plain (ny, nx) image or an
        // (nband, ny, nx) stack; ``psf_array`` and ``noise_array`` follow it,
        // and ``variance`` carries one value per band.  Several bands are
        // combined into one qimage after each band's PSF has been removed --
        // see prepare_data_cell_coadd in image.h.
        const std::vector<double> variance_in = to_variance_vector(variance);
        check_band_stack(img_array, psf_array, variance_in, noise_array);

        std::vector<double> variance_use = variance_in;
        if (noise_array.has_value()) {
            for (double& v : variance_use) {
                v = v * 2.0;
            }
        }

        // Band weights depend on the PSF, which varies from cell to cell, so
        // they are worked out per cell and handed to BOTH the detection and
        // the measurement.  Passing them explicitly is what guarantees the two
        // stages see the same coadd.
        auto cell_weights = [&](const py::array_t<double>& psf,
                                 const geometry::cell & cell) {
            return band_weights(
                cell.scale, this->sigma_arcsec_det, psf, variance_use
            );
        };

        std::vector<table::galNumber> catalog;
        if (detection.has_value()) {
            catalog = table::array_to_objlist(
                *detection
            );
        }

        this->fitter.do_fpfs = do_fpfs;

        const std::size_t ncells = cell_list.size();

        // External detections: recompute the owner of every row (cell_id
        // is derived state, never trusted from the input -- a stale value
        // would leave sources unmeasured) and bucket the rows by owner.
        // Internal detections need no assignment pass: find_peaks scans
        // each cell's half-open inner region, which is the same ownership
        // rule assign_cell_ids applies, so the detecting cell IS the
        // owner by construction.
        const bool external = detection.has_value();
        std::vector<std::vector<std::size_t>> rows_of(ncells);
        if (external) {
            assign_cell_ids(catalog, cell_list);
            std::unordered_map<int, std::size_t> pos_of;
            for (std::size_t ib = 0; ib < ncells; ++ib) {
                pos_of[cell_list[ib].index] = ib;
            }
            for (std::size_t i = 0; i < catalog.size(); ++i) {
                const auto it = pos_of.find(catalog[i].cell_id);
                if (it == pos_of.end()) {
                    // assign_cell_ids leaves cell_id = -1 only when the
                    // cell list is empty; a bare .at() would surface
                    // that as an opaque IndexError.
                    throw std::runtime_error(
                        "anacal Error: no cell owns the detection at "
                        "index " + std::to_string(i) +
                        " (is the cell list empty?)"
                    );
                }
                rows_of[it->second].push_back(i);
            }
        }

        // Flux error of the coadd: each band contributes its own Gaussian
        // flux variance, combined with the same w_b^2 weighting used for
        // the image itself.  One band reduces to the previous expression.
        auto compute_flux_errors = [&](const py::array_t<double>& psf,
                                       const std::vector<double>& w) {
            double var0 = 0.0;
            double var2 = 0.0;
            for (std::size_t b = 0; b < w.size(); ++b) {
                const ssize_t ib = static_cast<ssize_t>(b);
                const double ww = w[b] * w[b] * variance_use[b];
                var0 = var0 + ww * std::max(
                    0.0, gaussian_flux_variance_band(
                        psf, ib, 0.0, this->sigma_arcsec, this->scale
                    )
                );
                var2 = var2 + ww * std::max(
                    0.0, gaussian_flux_variance_band(
                        psf, ib, 0.2, this->sigma_arcsec, this->scale
                    )
                );
            }
            std::array<double, 2> errs{};
            errs[0] = std::sqrt(var0);
            errs[1] = std::sqrt(var2);
            return errs;
        };

        // ONE loop over cells.  Each iteration computes what its cell
        // needs (band weights and coadd variances run an FFT pipeline per
        // band, so they are computed once here and shared by detection,
        // flux errors and measurement), takes the sources the cell OWNS
        // -- detected in its inner region, or gathered from the external
        // catalog -- and measures exactly those.  Cells are fully
        // independent of each other.
        for (std::size_t ib = 0; ib < ncells; ++ib) {
            const geometry::cell & cell = cell_list[ib];
            const py::array_t<double>& psf = choose_psf(cell, psf_array);
            const std::vector<double> w = cell_weights(psf, cell);
            const double var_det = coadd_smoothed_variance(
                cell.scale,
                this->sigma_arcsec_det,
                psf,
                variance_use,
                w
            );

            std::vector<table::galNumber> rows;
            if (external) {
                rows.reserve(rows_of[ib].size());
                for (std::size_t idx : rows_of[ib]) {
                    rows.push_back(catalog[idx]);
                }
            } else {
                rows = detect_cell(
                    img_array,
                    psf,
                    variance_use,
                    cell,
                    noise_array,
                    w,
                    var_det
                );
                for (table::galNumber& src : rows) {
                    src.model.a1 = math::qnumber(a_ini);
                    src.model.a2 = math::qnumber(a_ini);
                }
                // Stamp the mask value right after detection (pure C++
                // reads and a local kernel, so it stays inside the GIL
                // release).  External catalogs are NOT restamped: they
                // keep the mask_value assigned when they were detected.
                if (mask_array.has_value()) {
                    mask::add_pixel_mask_column(
                        rows,
                        *mask_array,
                        this->sigma_arcsec_det * 1.5,
                        this->scale
                    );
                }
            }

            if (do_measure && !rows.empty()) {
                // The flux errors are a property of the cell's PSF and
                // noise, NOT of the individual measurement, so they are
                // stamped on every row -- including sources the mask
                // cut skips.  Leaving them at zero there would turn any
                // downstream flux/flux_err into 0/0.
                const std::array<double, 2> cell_flux_errs =
                    compute_flux_errors(psf, w);
                for (table::galNumber& src : rows) {
                    src.flux_gauss0_err = cell_flux_errs[0];
                    src.flux_gauss2_err = cell_flux_errs[1];
                }
                const double var_meas = coadd_smoothed_variance(
                    cell.scale,
                    this->sigma_arcsec,
                    psf,
                    variance_use,
                    w
                );
                measure_cell(
                    rows,
                    img_array,
                    psf,
                    variance_use,
                    cell,
                    noise_array,
                    w,
                    var_meas,
                    var_det,
                    mask_value_max
                );
            }

            if (external) {
                // Scatter the measured rows back so the output keeps the
                // input row order.
                for (std::size_t k = 0; k < rows_of[ib].size(); ++k) {
                    catalog[rows_of[ib][k]] = std::move(rows[k]);
                }
            } else {
                catalog.reserve(catalog.size() + rows.size());
                for (table::galNumber& src : rows) {
                    catalog.push_back(std::move(src));
                }
            }
        }

        // Reacquire the GIL: the output conversion below allocates a
        // numpy array.
        release_for_cells.reset();

        return table::objlist_to_array(catalog);
    };
};

void pyExportTask(py::module_& m);

} // task
} // anacal

#endif // ANACAL_TASK_H
