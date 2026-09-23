#ifndef ANACAL_NGMIX_FITTING_H
#define ANACAL_NGMIX_FITTING_H

#include "../image.h"
#include "../math.h"
#include "../table.h"


namespace anacal {
namespace ngmix {


// Radius of the model-fitting window about each source, in arcsec.
inline constexpr double fit_radius_arcsec = 3.5;

// Smallest semi-axis the moment-based initialisation hands to the fit,
// in arcsec.  It must be strictly positive: every a1 / a2 derivative of
// the model carries a factor a1 / a2 (the model depends on their
// squares), so a source started at exactly zero would never move.
inline constexpr double init_a_min = 0.1;

class GaussFit {
public:
    // stamp dimension
    double scale;
    double sigma_arcsec;
    int stamp_size, ss2;
    bool force_size, force_center;
    double fpfs_c0;
    bool do_fpfs;
    // damping of the shape / centre step (see update_model_params):
    // lam_k = lm_lambda0 * lm_decay^k on epoch k, a constant floor on
    // the diagonal, a misfit term, and per-epoch trust radii on the
    // shape (arcsec^2) and the centre (arcsec)
    double lm_lambda0, lm_decay, damping_floor, damping_rel;
    double trust_shape, trust_center, misfit_damping;
    // smooth convergence gate on the chi2 decrease achieved by the last
    // step, RELATIVE to the source's own chi2 scale F^2 H_FF / 2 (the
    // chi2 of the model against an empty image, ~ (S/N)^2), so that the
    // tolerance means the same relative precision for a faint and a
    // bright source (rmodel.h gate_factor): 0 at conv_tol, 1 at conv_tol
    // * gate_ratio; gate_ratio <= 1 makes it a hard step.  DEFAULT 0:
    // the gate is off and every source takes num_epochs epochs.  On a
    // real coadd the gate's own derivative term, -(ds/dc)(dc/dg) step,
    // is heavy-tailed on the sources that pass through the ramp
    // without stopping (one DP1 source went from R = +4.6 to -1338 at
    // a relative tolerance of 1e-6, and a wider ramp does not help),
    // while at a tolerance harmless for the response it stops almost
    // nobody; with the relative damping the fit converges in 5-8 epochs
    // anyway.  Kept for tests and experiments.
    double conv_tol, gate_ratio;
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
        bool do_fpfs=true,
        double lm_lambda0=0.2,
        double lm_decay=0.5,
        double damping_floor=1.0,
        double damping_rel=0.1,
        double trust_shape=0.05,
        double trust_center=0.1,
        double misfit_damping=1.0,
        double conv_tol=0.0,
        double gate_ratio=10.0
    ) : scale(scale), sigma_arcsec(sigma_arcsec), stamp_size(stamp_size),
        ss2(stamp_size / 2), force_size(force_size),
        force_center(force_center),
        fpfs_c0(fpfs_c0),
        do_fpfs(do_fpfs),
        lm_lambda0(lm_lambda0), lm_decay(lm_decay),
        damping_floor(damping_floor), damping_rel(damping_rel),
        trust_shape(trust_shape), trust_center(trust_center),
        misfit_damping(misfit_damping),
        conv_tol(conv_tol), gate_ratio(gate_ratio)
    {
        if (!(conv_tol >= 0.0) || !(gate_ratio >= 0.0)) {
            throw std::invalid_argument(
                "GaussFit: need conv_tol >= 0 and gate_ratio >= 0"
            );
        }
        if (trust_shape < 0.0 || trust_center < 0.0 || misfit_damping < 0.0) {
            throw std::invalid_argument(
                "GaussFit: need trust_shape, trust_center, misfit_damping >= 0"
            );
        }
        if (lm_lambda0 < 0.0 || lm_decay <= 0.0 || lm_decay > 1.0 ||
            damping_floor < 0.0 || damping_rel < 0.0) {
            throw std::invalid_argument(
                "GaussFit: need lm_lambda0 >= 0, 0 < lm_decay <= 1, "
                "damping_floor >= 0, damping_rel >= 0"
            );
        }
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
                    if (r2.v.v >= apod_r2_hi) {
                        // model and all its derivatives are exactly
                        // zero here: only the data term of chi2 is
                        // left, kept so that loss.v stays the chi2 of
                        // the whole window
                        const math::qnumber& d = data[jj + i];
                        src.loss.v = src.loss.v + d * d * (1.0 / variance);
                        continue;
                    }
                    src.loss = src.loss + model.get_loss(
                        data[jj + i], variance, r2, kernel
                    );
                }
            }
        }
        return;
    };

    // Flux pre-pass: F = sum(d m~) / (sum(m~^2) + w_F var / 2) for the
    // CURRENT shape, the exact minimiser of chi2 (+ the Gaussian flux
    // prior) in F.  Called at the top of every epoch, before the loss and
    // the shape step, so the shape gradients are always taken at the
    // optimal flux for the shape they belong to: d chi2 / dF = 0 there,
    // so the partial shape gradient at fixed F is the gradient of the
    // profiled chi2.  A negative or zero starting flux is overwritten
    // before any shape gradient is formed, and F* < 0 only when the data
    // in the window are genuinely negative.  F* is a ratio of two qnumber
    // sums, so the shear response propagates through it.
    inline void
    solve_flux(
        const std::vector<math::qnumber> & data,
        double variance,
        table::galNumber & src,
        const geometry::cell & cell,
        const modelPrior & prior
    ) const {
        ngmix::NgmixGaussian & model = src.model;
        const modelKernelB kb = model.prepare_modelB(
            this->scale, this->sigma_arcsec
        );
        const StampBounds bb = model.get_stamp_bounds(
            cell, fit_radius_arcsec / cell.scale
        );
        math::qnumber sdm, smm;
        for (int j = bb.j_min; (j < bb.j_max); ++j) {
            if (!cell.ymsk[j]) continue;
            int jj = j * cell.nx;
            for (int i = bb.i_min; (i < bb.i_max); ++i) {
                if (!cell.xmsk[i]) continue;
                if (bb.has_point(i, j)) {
                    math::qnumber r2 = model.get_r2(
                        cell.xvs[i], cell.yvs[j], kb
                    );
                    if (r2.v >= apod_r2_hi) continue;
                    math::qnumber mt = model.get_unit_model_from_r2(r2, kb);
                    sdm = sdm + data[jj + i] * mt;
                    smm = smm + mt * mt;
                }
            }
        }
        math::qnumber denom = smm + prior.w_F * (0.5 * variance);
        if (denom.v > 0.0) {
            model.F = sdm / denom;
        }
        return;
    };

    // Reduced chi2 a perfect model reaches on this cell: the robust
    // variance of the (deconvolved, re-smoothed) data over the cell,
    // MAD^2, divided by the variance the loss assumes, and never below
    // 1 (data quieter than the nominal noise -- a noiseless test stamp
    // -- simply gives a reduced chi2 below 1 and no misfit).  Every 2nd
    // pixel in each direction is enough for a per-cell scalar.
    inline double
    misfit_reference(
        const std::vector<math::qnumber> & data,
        double variance,
        const geometry::cell & cell
    ) const {
        std::vector<double> v;
        v.reserve(static_cast<std::size_t>(cell.nx) * cell.ny / 4 + 1);
        for (int j = 0; j < cell.ny; j += 2) {
            if (!cell.ymsk[j]) continue;
            int jj = j * cell.nx;
            for (int i = 0; i < cell.nx; i += 2) {
                if (!cell.xmsk[i]) continue;
                v.push_back(data[jj + i].v);
            }
        }
        if (v.size() < 16 || !(variance > 0.0)) return 1.0;
        const std::size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        const double med = v[mid];
        for (double& x : v) x = std::abs(x - med);
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        const double mad = 1.4826 * v[mid];
        return std::max(mad * mad / variance, 1.0);
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

    // One eigenvalue ``lam`` of the aperture-weighted covariance of the
    // deconvolved, re-smoothed image -> the intrinsic variance along it.
    // The Gaussian aperture (width sigma) and the re-smoothing (sigma)
    // both enter: for a Gaussian source of observed variance
    // c = m + sigma^2 the weighted moment is lam = c sigma^2 / (c +
    // sigma^2).  Both inversions can fail on real data -- a star, noise
    // or a blended neighbour push lam past sigma^2 or m below zero --
    // so they are wrapped in SMOOTH clamps (no kink in the estimator):
    // lam is held inside (0.05, 0.9) sigma^2, which caps m at 8
    // sigma^2, and m gets a positive floor of init_a_min^2.  A source
    // whose moments are unusable therefore starts at init_a_min, no
    // worse than the fixed a_ini it used to start at.
    inline math::qnumber
    init_eigen(const math::qnumber& lam) const {
        const double s2 = this->sigma2;
        math::qnumber lam_c = smooth_min(lam, 0.9 * s2, 0.05 * s2);
        lam_c = smooth_max(lam_c, 0.05 * s2, 0.05 * s2);
        math::qnumber c = lam_c * s2 / (s2 - lam_c);
        const double amin2 = init_a_min * init_a_min;
        return smooth_max(c - s2, amin2, amin2);
    };

    // Initial intrinsic covariance from the Gaussian-weighted quadrupole
    // moments of the source: eigen-decompose the weighted covariance,
    // map each eigenvalue through init_eigen and rebuild M at the same
    // angle.  Everything is a qnumber, so the shear response of the
    // starting point propagates like the rest of the fit.
    inline void
    initialize_shape(
        const std::vector<math::qnumber> & data,
        NgmixGaussian & model,
        const geometry::cell & cell
    ) const {
        math::qnumber m0, mxx, myy, mxy;
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
                    m0 = m0 + f;
                    mxx = mxx + f * x2;
                    myy = myy + f * y2;
                    mxy = mxy + f * xy;
                }
            }
        }
        if (m0.v > 0.0) {
            math::qnumber sxx = mxx / m0;
            math::qnumber syy = myy / m0;
            math::qnumber sxy = mxy / m0;
            math::qnumber tr = sxx + syy;
            // the tiny offset keeps sqrt differentiable for an exactly
            // round source (1e-8 arcsec^4, far below any real moment)
            math::qnumber diff = math::sqrt(
                math::pow(sxx - syy, 2) + 4.0 * math::pow(sxy, 2) + 1.0e-16
            );
            math::qnumber ang = 0.5 * math::atan2(2.0 * sxy, sxx - syy);
            model.set_eigen(
                this->init_eigen(0.5 * (tr + diff)),
                this->init_eigen(0.5 * (tr - diff)),
                ang
            );
        }
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
            // Same as initialize_shape: full qnumber subtraction keeps the
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
            src.model.sigma2_shape = this->sigma2;
            src.converged = false;
            src.n_epochs = 0;
            src.chi2_prev = math::qnumber(0.0);
            if (!src.initialized) {
                // Shape from the moments (replaces the a_ini / row
                // values unless the size is forced).  There is no
                // starting flux: the flux is profiled, solve_flux sets
                // it from the shape at the top of every epoch, and once
                // more below when there are no epochs.
                if (!this->force_size) {
                    initialize_shape(data, src.model, cell);
                }
                src.initialized = true;
            }
        }
        if (num_epochs == 0) {
            // No fit: the exported flux is still the profiled model flux
            // for the shape and centre the source carries -- the moment
            // initialisation, a forced a_ini, or an input catalog's
            // covariance and centre.  With force_size and force_center
            // this is forced photometry with a Gaussian profile: the
            // flux (a qnumber, with its shear and shift responses) of
            // the given covariance at the given position.
            for (std::size_t i=0; i<ng; ++i) {
                table::galNumber & src = catalog[i];
                if (src.n_mask_base > mvmax) continue;
                this->solve_flux(data, variance_meas, src, cell, prior);
            }
        }

        const double misfit_ref = (this->misfit_damping > 0.0 && num_epochs > 0)
            ? this->misfit_reference(data, variance_meas, cell)
            : 1.0;
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            const double lam = this->lm_lambda0 * std::pow(
                this->lm_decay, static_cast<double>(epoch)
            );
            for (std::size_t i=0; i<ng; ++i) {
                table::galNumber & src = catalog[i];
                if (src.n_mask_base > mvmax) continue;
                if (src.converged) continue;
                this->solve_flux(data, variance_meas, src, cell, prior);
                const modelKernelD kernel = src.model.prepare_modelD(
                    this->scale,
                    this->sigma_arcsec
                );
                this->measure_loss(
                    data, variance_meas, src, cell, kernel
                );
                // gate on |chi2_prev - chi2|, the decrease the last step
                // achieved (open on the first epoch and when the gate is
                // disabled); |.| has its kink at 0, inside the region
                // where the gate is identically 0, so the gate stays smooth
                math::qnumber gate(1.0);
                if (this->conv_tol > 0.0 && src.n_epochs > 0) {
                    math::qnumber dchi2 = src.chi2_prev - src.loss.v;
                    if (dchi2.v < 0.0) dchi2 = -1.0 * dchi2;
                    // relative to the chi2 of the model itself
                    const math::qnumber& F = src.model.F;
                    math::qnumber scl = 0.5 * F * F * src.loss.v_FF;
                    if (scl.v > 0.0) {
                        gate = gate_factor(
                            dchi2 / scl, this->conv_tol,
                            this->conv_tol * this->gate_ratio
                        );
                    }
                }
                src.chi2_prev = src.loss.v;
                src.model.update_model_params(
                    src.loss, prior, src.x1_det, src.x2_det,
                    lam, this->damping_floor, this->damping_rel,
                    this->trust_shape, this->trust_center,
                    this->misfit_damping, misfit_ref, this->sigma2,
                    gate
                );
                if (src.n_epochs < 255) src.n_epochs += 1;
                // gate exactly zero: nothing moved and, chi2 being
                // unchanged from here on, nothing will; skip the source
                if (gate.v <= 0.0) src.converged = true;
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
