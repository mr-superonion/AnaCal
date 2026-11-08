#ifndef ANACAL_TASK_H
#define ANACAL_TASK_H

#include "detector.h"
#include "renoise.h"
#include "mask.h"

namespace anacal {
namespace task {

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
        int num_epochs=10,
        bool force_size=false,
        bool force_center=true,
        double fpfs_c0=1.0,
        double mag_zero=THRESHOLD_REF_MAG_ZERO,
        double lm_lambda0=0.2,
        double lm_decay=0.5,
        double damping_floor=1.0,
        double damping_rel=0.1,
        double trust_shape=0.05,
        double trust_center=0.1,
        double misfit_damping=1.0,
        double conv_tol=0.0,
        double gate_ratio=10.0
    ) : scale(scale), sigma_arcsec(sigma_arcsec), snr_peak_min(snr_peak_min),
        omega_f(omega_f), omega_v(omega_v),
        prior(prior ? *prior : ngmix::modelPrior()),
        stamp_size(stamp_size), image_bound(image_bound),
        num_epochs(num_epochs), fitter(
            scale, sigma_arcsec, stamp_size,
            force_size, force_center,
            fpfs_c0 * std::pow(
                10.0, (mag_zero - THRESHOLD_REF_MAG_ZERO) / 2.5
            ),
            true, lm_lambda0, lm_decay, damping_floor, damping_rel,
            trust_shape, trust_center, misfit_damping, conv_tol, gate_ratio
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
        const std::optional<py::array_t<pixel_t>>& noise2_array=std::nullopt,
        const std::optional<std::vector<double>>& weights=std::nullopt,
        const std::optional<double>& multiband_coadd_variance=std::nullopt,
        const std::vector<renoise::RenoiseKernels>* rks=nullptr
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
            noise2_array,
            this->image_bound,
            weights,
            multiband_coadd_variance,
            rks
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
        const std::optional<py::array_t<pixel_t>>& noise2_array=std::nullopt,
        const std::optional<std::vector<double>>& weights=std::nullopt,
        const std::optional<double>& variance_meas=std::nullopt,
        const std::optional<double>& n_mask_base_max=std::nullopt,
        const std::vector<renoise::RenoiseKernels>* rks=nullptr
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
            noise2_array,
            weights,
            variance_meas,
            n_mask_base_max,
            rks
        );
        for (table::galNumber & src : rows) {
            src.decentralize(cell);
        }
        return;
    };

    // The detection + measurement of process_image on C++ containers.
    inline std::vector<table::galNumber>
    process_image_impl(
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        const varianceArg& variance,
        const std::vector<geometry::cell>& cell_list,
        const std::optional<py::array_t<table::galRow>>& detection,
        const std::optional<py::array_t<pixel_t>>& noise_array,
        const std::optional<py::array_t<pixel_t>>& noise2_array,
        const std::optional<py::array_t<uint8_t>>& mask_array,
        double a_ini,
        bool do_measure,
        bool do_fpfs,
        const std::optional<double>& n_mask_base_max,
        const renoise::Setup* renoise
    ) {
        // Everything here works on C++ containers and READS the input
        // arrays through unchecked accessors only -- the prologue below
        // (variance normalization, band-stack validation, weights lambda,
        // detection conversion) is pure C++ plus GIL-free ndim/shape
        // struct reads, and validation errors unwind safely through the
        // release -- so the GIL is dropped for the whole call; this is
        // what lets callers thread over cells.  (The FFT engine is
        // lock-free: see image.h.)  The callers convert the result to
        // numpy after the release has ended.
        ScopedGilRelease release_for_cells;

        // ``img_array`` may be a plain (ny, nx) image or an
        // (nband, ny, nx) stack; ``psf_array`` and ``noise_array`` follow it,
        // and ``variance`` carries one value per band.  Several bands are
        // combined into one qimage after each band's PSF has been removed --
        // see prepare_data_cell_coadd in image.h.
        const std::vector<double> variance_in = to_variance_vector(variance);
        check_band_stack(
            img_array, psf_array, variance_in, noise_array, noise2_array
        );

        // A renoise Setup carries the noise spectrum of the patch: the
        // variances are then summed over the actual total noise spectrum
        // in BOTH modes; the spectral kernels (filler, added noise,
        // assigned response) are applied to the measurement in Option 2
        // only.
        const bool use_spectra = (renoise != nullptr);
        const bool spectral = use_spectra
            && (renoise->mode == renoise::Mode::spectral);
        if (use_spectra && !noise_array.has_value()) {
            throw std::runtime_error(
                "anacal Error: renoising with a noise spectrum needs a "
                "noise image"
            );
        }
        if (spectral && !noise2_array.has_value()) {
            throw std::runtime_error(
                "anacal Error: spectral renoising needs a second noise "
                "image (the filler realization eps'')"
            );
        }
        // Option 1 doubles the noise variance everywhere; Option 2's factor
        // depends on the cell PSF and the smoothing scale and is worked out
        // per cell below.
        std::vector<double> variance_use = variance_in;
        if (noise_array.has_value() && !use_spectra) {
            for (double& v : variance_use) {
                v = v * 2.0;
            }
        }

        // Band weights depend on the PSF, which varies from cell to cell, so
        // they are worked out per cell and handed to BOTH the detection and
        // the measurement.  Passing them explicitly is what guarantees the two
        // stages see the same coadd.
        auto cell_weights = [&](const py::array_t<double>& psf,
                                 const geometry::cell & cell,
                                 const std::vector<double>& variance_det) {
            return band_weights(
                cell.scale, this->sigma_arcsec_det, psf, variance_det
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
            // Option 2: kernels of this cell's PSF and the matching noise
            // variances at the detection and measurement scales
            std::vector<renoise::RenoiseKernels> rks;
            const std::vector<renoise::RenoiseKernels>* rks_ptr = nullptr;
            std::vector<double> variance_det = variance_use;
            std::vector<double> variance_meas = variance_use;
            if (use_spectra) {
                rks = renoise::make_cell_kernels(
                    *renoise, psf, klim_pix / cell.scale
                );
                if (spectral) {
                    rks_ptr = &rks;
                }
                // equivalent white-noise variances of the smoothed image at
                // the detection and the measurement scale (the flux errors
                // are summed over the spectrum directly, see above)
                for (std::size_t b = 0; b < rks.size(); ++b) {
                    variance_det[b] = variance_in[b] * renoise::variance_factor(
                        rks[b], this->sigma_arcsec_det
                    );
                    variance_meas[b] = variance_in[b] * renoise::variance_factor(
                        rks[b], this->sigma_arcsec
                    );
                }
            }
            const std::vector<double> w = cell_weights(psf, cell, variance_det);

            std::vector<table::galNumber> rows;
            if (external) {
                rows.reserve(rows_of[ib].size());
                for (std::size_t idx : rows_of[ib]) {
                    rows.push_back(catalog[idx]);
                }
            } else {
                // Only detection needs the variance at the detection
                // scale; the measurement no longer uses it, so this stays
                // inside the branch that detects.
                rows = detect_cell(
                    img_array,
                    psf,
                    variance_det,
                    cell,
                    noise_array,
                    noise2_array,
                    w,
                    coadd_smoothed_variance(
                        cell.scale,
                        this->sigma_arcsec_det,
                        psf,
                        variance_det,
                        w
                    ),
                    rks_ptr
                );
                for (table::galNumber& src : rows) {
                    src.model.set_axes(
                        math::qnumber(a_ini), math::qnumber(a_ini),
                        math::qnumber(0.0)
                    );
                }
                // Stamp the mask value right after detection (pure C++
                // reads and a local kernel, so it stays inside the GIL
                // release).  External catalogs are NOT restamped: they
                // keep the n_mask_base assigned when they were detected.
                if (mask_array.has_value()) {
                    mask::add_mask_fraction_columns(
                        rows,
                        *mask_array,
                        this->sigma_arcsec_det * 1.5,
                        this->scale
                    );
                }
            }

            if (do_measure && !rows.empty()) {
                const double var_meas = coadd_smoothed_variance(
                    cell.scale,
                    this->sigma_arcsec,
                    psf,
                    variance_meas,
                    w
                );
                measure_cell(
                    rows,
                    img_array,
                    psf,
                    variance_meas,
                    cell,
                    noise_array,
                    noise2_array,
                    w,
                    var_meas,
                    n_mask_base_max,
                    rks_ptr
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
        return catalog;
    };

    inline py::array_t<table::galRow>
    process_image(
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        const varianceArg& variance,
        const std::vector<geometry::cell>& cell_list,
        const std::optional<py::array_t<table::galRow>>& detection=std::nullopt,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
        // second, independent noise realization eps'' (same correlation
        // properties as the image noise): the spectral renoising builds
        // the spin-2 filler from it, in Fourier space, per cell
        const std::optional<py::array_t<pixel_t>>& noise2_array=std::nullopt,
        const std::optional<py::array_t<uint8_t>>& mask_array=std::nullopt,
        double a_ini=0.2,
        bool do_measure=true,
        bool do_fpfs=true,
        // Sources whose n_mask_base exceeds this are SKIPPED by the
        // measurement (rows kept with default values); internal
        // detections are stamped from mask_array before the check,
        // external catalogs carry their own n_mask_base column.
        const std::optional<double>& n_mask_base_max=std::nullopt,
        // Minimal-variance renoising (Option 2): per-band noise spectra
        // of the patch plus (c, delta).  Null: the rotated-noise
        // renoising (Option 1), whenever noise_array is given.
        const renoise::Setup* renoise=nullptr
    ) {
        const std::vector<table::galNumber> catalog = process_image_impl(
            img_array, psf_array, variance, cell_list, detection,
            noise_array, noise2_array, mask_array, a_ini, do_measure,
            do_fpfs, n_mask_base_max, renoise
        );
        return table::objlist_to_array(catalog);
    };
};

void pyExportTask(py::module_& m);

} // task
} // anacal

#endif // ANACAL_TASK_H
