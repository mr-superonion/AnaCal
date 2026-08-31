#ifndef ANACAL_NGMIX_FITTING_H
#define ANACAL_NGMIX_FITTING_H

#include "../image.h"
#include "../math.h"
#include "../table.h"


namespace anacal {
namespace ngmix {


// Radius of the model-fitting window about each source, in arcsec.
inline constexpr double fit_radius_arcsec = 3.5;

class GaussFit {
public:
    // stamp dimension
    double scale;
    double sigma_arcsec;
    int stamp_size, ss2;
    bool force_size, force_center;
    double fpfs_c0;
    bool do_fpfs;
    double sigma2, sigma_m2, rfac, ffac, ffac2, ffac3;
    double sigma2_lim;
    double r2_lim_stamp;

    GaussFit(
        double scale,
        double sigma_arcsec,
        int stamp_size=64,
        bool force_size=false,
        bool force_center=false,
        double fpfs_c0=1.0,
        bool do_fpfs=true
    ) : scale(scale), sigma_arcsec(sigma_arcsec), stamp_size(stamp_size),
        ss2(stamp_size / 2), force_size(force_size),
        force_center(force_center),
        fpfs_c0(fpfs_c0),
        do_fpfs(do_fpfs)
    {
        this->sigma2 = sigma_arcsec * sigma_arcsec;
        this->sigma_m2 = 1.0 / this->sigma2;
        this->rfac = -0.5 * this->sigma_m2;
        this->ffac = rfac * (-0.318309886);
        this->ffac2 = this->ffac * sqrt2 * this->sigma_m2;
        this->ffac3 = this->ffac * 2.0 * this->sigma_m2;
        this->sigma2_lim = sigma2 * 20;
        this->r2_lim_stamp = std::pow((this->ss2-1) * scale, 2.0);
    };

    inline void
    measure_loss(
        const std::vector<math::qnumber> & data,
        double variance,
        table::galNumber & src,
        const geometry::cell & cell,
        const modelKernelD & kernel
    ) const {
        src.loss.reset();
        ngmix::NgmixGaussian & model = src.model;
        const StampBounds bb = model.get_stamp_bounds(cell, fit_radius_arcsec / cell.scale);

        for (int j = bb.j_min; (j < bb.j_max); ++j) {
            if (!cell.ymsk[j]) continue;
            int jj = j * cell.nx;
            for (int i = bb.i_min; (i < bb.i_max); ++i) {
                if (!cell.xmsk[i]) continue;
                if (bb.has_point(i, j)) {
                    math::lossNumber r2 = model.get_r2(
                        cell.xvs[i], cell.yvs[j], kernel
                    );
                    src.loss = src.loss + model.get_loss(
                        data[jj + i], variance, r2, kernel
                    );
                }
            }
        }
        return;
    };

    inline void
    measure_gaussian_fluxes(
        const std::vector<math::qnumber> & data,
        table::galNumber & src,
        const geometry::cell & cell
    ) const {
        const double sigma0_2 = this->sigma2 + 0 * 0;
        const double sigma2_2 = this->sigma2 + 0.2 * 0.2;
        src.flux_gauss0 = measure_flux(sigma0_2, cell, data, src.model);
        src.flux_gauss2 = measure_flux(sigma2_2, cell, data, src.model);
        return;
    };

    inline void
    measure_fpfs(
        const std::vector<math::qnumber> & data,
        table::galNumber & src,
        const geometry::cell & cell
    ) const {
        ngmix::NgmixGaussian & model = src.model;
        int r = static_cast<int>(this->sigma_arcsec * 8 / cell.scale);
        // Clip the window about the source centre; clamping i_max against
        // i_min (instead of i_cen) would shift the whole window inward at a
        // cell edge, making the aperture asymmetric about the source.
        int i_cen = static_cast<int>(
            std::round(model.x1.v / this->scale)
        ) - cell.xmin;
        int i_min = std::max(i_cen - r, 0);
        int i_max = std::min(i_cen + r + 1, cell.nx);
        int j_cen = static_cast<int>(
            std::round(model.x2.v / this->scale)
        ) - cell.ymin;
        int j_min = std::max(j_cen - r, 0);
        int j_max = std::min(j_cen + r + 1, cell.ny);

        math::qnumber m0, mxx, myy, mxy;
        for (int j = j_min; j < j_max; ++j) {
            if (!cell.ymsk[j]) continue;
            int jj = j * cell.nx;
            double ys = cell.yvs[j] - model.x2.v;
            double y2 = ys * ys;
            for (int i = i_min; i < i_max; ++i) {
                if (!cell.xmsk[i]) continue;
                double xs = cell.xvs[i] - model.x1.v;
                double x2 = xs * xs;
                if ((x2 + y2) < this->sigma2_lim) {
                    std::array<math::qnumber, 4> mm = src.model.get_fpfs_moments(
                        data[jj + i],
                        cell.xvs[i],
                        cell.yvs[j],
                        this->rfac
                    );
                    m0 = m0 + mm[0];
                    mxx = mxx + mm[1];
                    myy = myy + mm[2];
                    mxy = mxy + mm[3];
                }
            }
        }
        src.fpfs_m0 = m0 * this->ffac;
        src.fpfs_m2 = (mxx + myy - m0 * this->sigma2) * this->ffac3;
        {
            math::qnumber denom = (src.fpfs_m0 + this->fpfs_c0);
            src.fpfs_e1 = (mxx - myy) * this->ffac2 / denom;
            src.fpfs_e2 = 2.0 * mxy * this->ffac2 / denom;
        }
        return;
    };

    inline void
    initialize_angle(
        const std::vector<math::qnumber> & data,
        NgmixGaussian & model,
        const geometry::cell & cell
    ) const {
        math::qnumber mxx, myy, mxy;
        double dd = 1.0 / this->sigma2;

        const StampBounds bb = model.get_stamp_bounds(cell, fit_radius_arcsec / cell.scale);
        for (int j = bb.j_min; (j < bb.j_max); ++j) {
            if (!cell.ymsk[j]) continue;
            int jj = j * cell.nx;
            // Full qnumber subtraction: the aperture follows the fitted
            // centroid, so the centroid's shear response must propagate
            // (the initial centre is a grid point with zero response).
            math::qnumber ys = cell.yvs[j] - model.x2;
            math::qnumber y2 = math::pow(ys, 2);
            for (int i = bb.i_min; (i < bb.i_max); ++i) {
                if (!cell.xmsk[i]) continue;
                math::qnumber xs = cell.xvs[i] - model.x1;
                math::qnumber x2 = math::pow(xs, 2);
                math::qnumber xy = xs * ys;
                math::qnumber r2 = (x2 + y2) * dd;
                if (bb.has_point(i, j)) {
                    math::qnumber w = math::exp(-0.5 * r2);
                    math::qnumber f = w * data[jj + i];
                    mxx = mxx + f * x2;
                    myy = myy + f * y2;
                    mxy = mxy + f * xy;
                }
            }
        }
        model.t = 0.5 * math::atan2(2.0 * mxy, mxx - myy);
        return;
    };

    inline math::qnumber
    measure_flux(
        double sigma_meas2,
        const geometry::cell & cell,
        const std::vector<math::qnumber> & data,
        const NgmixGaussian & model
    ) const {
        math::qnumber m0, norm;
        double dd = 1.0 / sigma_meas2;

        const StampBounds bb = model.get_stamp_bounds(cell, fit_radius_arcsec / cell.scale);
        for (int j = bb.j_min; (j < bb.j_max); ++j) {
            if (!cell.ymsk[j]) continue;
            int jj = j * cell.nx;
            // Same as initialize_angle: full qnumber subtraction keeps the
            // centroid's shear response in the aperture weight, matching
            // get_fpfs_moments (rmodel.h).
            math::qnumber ys = cell.yvs[j] - model.x2;
            math::qnumber y2 = math::pow(ys, 2);
            for (int i = bb.i_min; (i < bb.i_max); ++i) {
                if (!cell.xmsk[i]) continue;
                math::qnumber xs = cell.xvs[i] - model.x1;
                math::qnumber x2 = math::pow(xs, 2);
                math::qnumber r2 = (x2 + y2) * dd;
                if (bb.has_point(i, j)) {
                    math::qnumber w = math::exp(-0.5 * r2);
                    math::qnumber f = w * data[jj + i];
                    norm = norm + w * w;
                    m0 = m0 + f;
                }
            }
        }

        if (norm.v > 0.0) {
            math::qnumber flux = m0 * (2.0 * M_PI * sigma_meas2);
            flux = flux / norm;
            double scale2 = cell.scale * cell.scale;
            flux = flux / scale2;
            return flux;
        }

        return math::qnumber(0.0);
    };

    inline void
    initialize_flux(
        const std::vector<math::qnumber> & data,
        NgmixGaussian & model,
        const geometry::cell & cell
    ) const {
        double a_sum = model.a1.v + model.a2.v;
        double sigma2_flux = this->sigma2 + 0.25 * a_sum * a_sum;
        model.F = measure_flux(sigma2_flux, cell, data, model);
        return;
    };

    inline void
    process_cell_impl(
        std::vector<table::galNumber>& catalog,
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        const modelPrior & prior,
        int num_epochs,
        const std::vector<double>& variance,
        const geometry::cell & cell,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
        const std::optional<std::vector<double>>& weights=std::nullopt,
        const std::optional<double>& variance_meas_opt=std::nullopt,
        const std::optional<double>& n_mask_base_max=std::nullopt
    ) {
        // PRECONDITION: the band stacks were validated by the caller
        // (Task::process_image or process_cell below) -- validating once
        // per public entry point instead of once per layer.
        // PRECONDITION: ``catalog`` holds exactly the sources this cell
        // measures (Task::process_image hands each cell its OWN sources
        // only; process_cell below hands over the caller's whole catalog).
        // Every row is processed -- there is no ownership guard here.
        // The bands must be combined here exactly as they were for detection,
        // so the weights come from the caller.  Only when this is used
        // stand-alone (process_cell below) are they derived here, and then at
        // the DETECTION scale, sigma * sqrt2, to match.
        std::vector<double> w;
        if (weights.has_value()) {
            if (weights->size() != variance.size()) {
                throw std::runtime_error(
                    "ngmix Error: got " + std::to_string(weights->size()) +
                    " band weights for " + std::to_string(variance.size()) +
                    " band(s)"
                );
            }
            w = *weights;
        } else {
            w = band_weights(
                cell.scale, detection_sigma(this->sigma_arcsec),
                psf_array, variance
            );
        }

        std::vector<math::qnumber> data = prepare_data_cell_coadd(
            img_array,
            psf_array,
            this->sigma_arcsec,
            cell,
            w,
            noise_array
        );

        const std::size_t ng = catalog.size();

        // Task::process_image precomputes both coadd variances once per
        // cell; the stand-alone process_cell below derives them here.
        double variance_meas = variance_meas_opt.has_value()
            ? *variance_meas_opt
            : coadd_smoothed_variance(
                cell.scale,
                this->sigma_arcsec,
                psf_array,
                variance,
                w
            );
        // Sources on heavily masked pixels are SKIPPED, not dropped:
        // their rows stay in the catalog with default measurement
        // values, flagged by their n_mask_base column.
        const float mvmax = n_mask_base_max.has_value()
            ? static_cast<float>(*n_mask_base_max)
            : std::numeric_limits<float>::max();

        // initialize the sources
        for (std::size_t i = 0; i < ng; ++i) {
            table::galNumber & src = catalog[i];
            if (src.n_mask_base > mvmax) {
                // Skipped sources must be INERT downstream: they carry
                // a real wsel (> 0 from detection, or stated by the
                // caller for a forced catalog) but never get measured,
                // so without this they would enter weighted sums with a
                // real selection weight and a default (zero) shape --
                // exactly what the fail-closed contract in table.h
                // forbids.
                src.wsel = math::qnumber();
                continue;
            }
            src.model.force_size=this->force_size;
            src.model.force_center=this->force_center;
            if (!src.initialized) {
                if (!this->force_size) {
                    initialize_angle(data, src.model, cell);
                }
                initialize_flux(data, src.model, cell);
                src.initialized = true;
            }
        }

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            for (std::size_t i=0; i<ng; ++i) {
                table::galNumber & src = catalog[i];
                if (src.n_mask_base > mvmax) continue;
                const modelKernelD kernel = src.model.prepare_modelD(
                    this->scale,
                    this->sigma_arcsec
                );
                this->measure_loss(
                    data, variance_meas, src, cell, kernel
                );
                src.model.update_model_params(
                    src.loss, prior, src.x1_det, src.x2_det, variance_meas
                );
            }
        }

        for (std::size_t i=0; i<ng; ++i) {
            table::galNumber & src = catalog[i];
            if (src.n_mask_base > mvmax) continue;
            this->measure_gaussian_fluxes(
                data, src, cell
            );
            // The measurement does NOT set wsel.  It is fixed at
            // detection (detector::measure_pixel), or stated by the
            // caller for a forced catalog; the FPFS size cut that used
            // to multiply into it here has been removed.
            if (this->do_fpfs) {
                this->measure_fpfs(
                    data, src, cell
                );
            }
        }
        return;
    };

    inline std::vector<table::galNumber>
    process_cell(
        const std::vector<table::galNumber>& catalog,
        const py::array_t<pixel_t>& img_array,
        const py::array_t<double>& psf_array,
        const modelPrior & prior,
        const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
        int num_epochs = 5,
        const varianceArg& variance = 1.0,
        std::optional<geometry::cell> cell=std::nullopt,
        const std::optional<double>& n_mask_base_max=std::nullopt
    ) {
        const ssize_t nd = img_array.ndim();
        int image_ny = static_cast<int>(img_array.shape(nd - 2));
        int image_nx = static_cast<int>(img_array.shape(nd - 1));
        geometry::cell bb = cell ? *cell : geometry::get_cell_list(
            image_nx, image_ny, image_nx, image_ny, 0, this->scale
        )[0];
        // The whole measurement is allocation-free (inputs are only read
        // through unchecked accessors; the catalog copy, cell_id stamping
        // and variance/band validation below are pure C++), so drop the
        // GIL unless a caller higher up already did.  Placed after the
        // ``bb`` copy above, whose ``psf_array`` refcount needs the GIL.
        ScopedGilRelease release;
        std::vector<table::galNumber> result = catalog;
        // The caller hands this function the catalog FOR this cell, so the
        // cell measures every source in it.  cell_id is derived state
        // (not a trusted input column): stamp it so the OUTPUT records
        // which cell did the measuring.
        for (table::galNumber & src : result) {
            src.cell_id = bb.index;
        }
        const std::vector<double> variance_vec = to_variance_vector(variance);
        check_band_stack(img_array, psf_array, variance_vec, noise_array);
        process_cell_impl(
            result,
            img_array,
            psf_array,
            prior,
            num_epochs,
            variance_vec,
            bb,
            noise_array,
            std::nullopt,
            std::nullopt,
            n_mask_base_max
        );
        return result;
    };
};

} // end of ngmix
} // end of anacal

#endif // ANACAL_NGMIX_FITTING
