#ifndef ANACAL_NGMIX_RMODEL_H
#define ANACAL_NGMIX_RMODEL_H

#include "../math.h"

namespace anacal {
namespace ngmix {

inline constexpr double one_over_two_pi = 0.5 / M_PI;

// Smooth clamps (C-infinity, exact in the limits) used where a hard
// max / min would put a kink in the estimator: smooth_max(y, lo) -> lo
// for y << lo and -> y for y >> lo, with the hand-over spread over
// ~eps; smooth_min the same from above.  Far from the clamp the
// deviation from the identity is eps^2 / (4 |y - lo|).
inline math::qnumber
smooth_max(const math::qnumber& y, double lo, double eps) {
    math::qnumber d = y - lo;
    return lo + 0.5 * (d + math::sqrt(d * d + eps * eps));
}

inline math::qnumber
smooth_min(const math::qnumber& y, double hi, double eps) {
    math::qnumber d = y - hi;
    return hi + 0.5 * (d - math::sqrt(d * d + eps * eps));
}

// Apodisation of the Gaussian in r^2 (r^2 = the quadratic form of the
// model): 1 up to apod_r2_lo, a C1 cubic ramp, exactly 0 from
// apod_r2_hi on.  Beyond apod_r2_hi the model AND every derivative are
// exactly zero, so a pixel there can be skipped by the fit with no
// approximation even though the edge moves with the parameters.  The
// same values as ngmix's fexp apodisation (chi2 20 .. 25); the
// Gaussian is already exp(-10) = 4.5e-5 of its peak at the inner edge.
inline constexpr double apod_r2_lo = 20.0;
inline constexpr double apod_r2_hi = 25.0;

struct apodDeriv {
    // A(r^2) and dA / d(r^2)
    math::qnumber a, da;
};

inline apodDeriv
get_apod(const math::qnumber& r2) {
    if (r2.v <= apod_r2_lo) {
        return {math::qnumber(1.0), math::qnumber(0.0)};
    }
    if (r2.v >= apod_r2_hi) {
        return {math::qnumber(0.0), math::qnumber(0.0)};
    }
    const double w = apod_r2_hi - apod_r2_lo;
    math::qnumber u = (r2 - apod_r2_lo) / w;
    math::qnumber u2 = u * u;
    return {
        1.0 - (3.0 * u2 - 2.0 * u2 * u),
        (6.0 * u2 - 6.0 * u) / w
    };
}

// Solve A x = b for x, with A symmetric positive definite and every
// entry a qnumber, n <= 5.  The values go through a Cholesky
// factorisation of A.v; the four derivative slots follow from
// differentiating A x = b:  A x.s = b.s - A.s x  (same factorisation,
// one back-substitution per slot).  Returns false, leaving x untouched,
// if A.v is not positive definite.
inline bool
solve_spd_qnumber(
    int n,
    const std::array<std::array<math::qnumber, 5>, 5>& A,
    const std::array<math::qnumber, 5>& b,
    std::array<math::qnumber, 5>& x
) {
    std::array<std::array<double, 5>, 5> L{};
    for (int j = 0; j < n; ++j) {
        double d = A[j][j].v;
        for (int k = 0; k < j; ++k) d -= L[j][k] * L[j][k];
        if (!(d > 0.0)) return false;
        L[j][j] = std::sqrt(d);
        for (int i = j + 1; i < n; ++i) {
            double s = A[i][j].v;
            for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
            L[i][j] = s / L[j][j];
        }
    }
    auto solve = [&](std::array<double, 5> rhs) {
        // forward L y = rhs, then backward L^T z = y (in place)
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < i; ++k) rhs[i] -= L[i][k] * rhs[k];
            rhs[i] /= L[i][i];
        }
        for (int i = n - 1; i >= 0; --i) {
            for (int k = i + 1; k < n; ++k) rhs[i] -= L[k][i] * rhs[k];
            rhs[i] /= L[i][i];
        }
        return rhs;
    };
    std::array<double, 5> rhs{};
    for (int i = 0; i < n; ++i) rhs[i] = b[i].v;
    const std::array<double, 5> xv = solve(rhs);
    // derivative slots: A x.s = b.s - A.s x
    std::array<std::array<double, 5>, 4> xs{};
    for (int s = 0; s < 4; ++s) {
        for (int i = 0; i < n; ++i) {
            double r = 0.0;
            switch (s) {
                case 0: r = b[i].g1; break;
                case 1: r = b[i].g2; break;
                case 2: r = b[i].x1; break;
                default: r = b[i].x2; break;
            }
            for (int j = 0; j < n; ++j) {
                double a = 0.0;
                switch (s) {
                    case 0: a = A[i][j].g1; break;
                    case 1: a = A[i][j].g2; break;
                    case 2: a = A[i][j].x1; break;
                    default: a = A[i][j].x2; break;
                }
                r -= a * xv[j];
            }
            rhs[i] = r;
        }
        xs[s] = solve(rhs);
    }
    for (int i = 0; i < n; ++i) {
        x[i] = math::qnumber(xv[i], xs[0][i], xs[1][i], xs[2][i], xs[3][i]);
    }
    return true;
}

// Smooth trust radius: scale a step vector of squared length d2 so that
// its length never exceeds r, leaving a step much shorter than r
// almost untouched (relative change (|step| / r)^2 / 2).
inline math::qnumber
trust_factor(const math::qnumber& d2, double r) {
    return r / math::sqrt(d2 + r * r);
}


struct modelPrior {
    // Gaussian priors, each stored as w = 2 / sigma^2 (the loss is a
    // chi2): w_F on the flux (towards 0), w_a on each of the three
    // intrinsic covariance components mxx, myy, mxy (towards 0; sigma
    // in arcsec^2), w_x on the centre (towards the detection position;
    // sigma in arcsec).
    math::qnumber w_F, w_a, w_x;

    modelPrior() = default;

    inline void set_sigma_F(math::qnumber sigma_F){
        this->w_F = 2.0 / math::pow(sigma_F, 2.0);
    };

    inline void set_sigma_a(math::qnumber sigma_a){
        this->w_a = 2.0 / math::pow(sigma_a, 2.0);
    };

    inline void set_sigma_x(math::qnumber sigma_x){
        this->w_x = 2.0 / math::pow(sigma_x, 2.0);
    };
};

// Inverse of the SMOOTHED covariance C = M + sigma^2 I, i.e. the
// quadratic form of the model, plus the normalisation f.
struct modelKernelB {
    math::qnumber ixx, ixy, iyy;
    math::qnumber f;
    double scale;

    modelKernelB() = default;
};

// The same plus the derivatives of I = C^{-1} and of f with respect to
// the three intrinsic covariance components: dI[p][k] for p in
// (mxx, myy, mxy) and k in (xx, xy, yy).
struct modelKernelD {
    math::qnumber ixx, ixy, iyy;
    std::array<std::array<math::qnumber, 3>, 3> dI;
    math::qnumber f, f_mxx, f_myy, f_mxy;
    double scale;

    modelKernelD() = default;
};

struct frDeriv {
    // f(r) and its first derivative
    math::qnumber fr, dfr;

    frDeriv() = default;

    frDeriv(
        math::qnumber fr, math::qnumber dfr
    )
        : fr(fr), dfr(dfr) {}
};


struct StampBounds {
    int i_min, i_max;
    int j_min, j_max;
    int i_cen, j_cen;
    int r;

    inline bool
    has_point(
        int i, int j
    ) const {
        int di = (i - i_cen);
        int dj = (j - j_cen);
        return  (di * di + dj * dj) < r * r;
    };
};

// A single elliptical Gaussian, fitted to the PSF-deconvolved image
// re-smoothed with an isotropic Gaussian of width sigma:
//     m(x) = F scale^2 / (2 pi sqrt(det C)) exp(-x^T C^{-1} x / 2)
//     C    = M + sigma^2 I
// The free parameters are the flux F, the centre (x1, x2) in arcsec
// and the INTRINSIC covariance M = (mxx, myy, mxy) in arcsec^2.  The
// covariance components are the natural parameters: the Gaussian is
// regular in them everywhere, including for a round source, whereas
// (semi-axes, angle) has no curvature in the angle at roundness and an
// angle response that diverges as the ellipticity goes to zero.  The
// semi-axes and the angle are still available as DERIVED quantities
// (get_axes / set_axes) for the catalog columns and older callers.
class NgmixGaussian {
private:
    frDeriv get_fr(
        math::qnumber r2
    ) const {
        // exp(-r2/2) * A(r2) and its r2-derivative; inside apod_r2_lo
        // this is bit-identical to the plain Gaussian (a = 1, da = 0)
        math::qnumber e = math::exp(r2 * (-0.5));
        apodDeriv ap = get_apod(r2);
        return frDeriv(e * ap.a, e * (ap.da - 0.5 * ap.a));
    };

    // eigen-decomposition of M: (lambda_plus, lambda_minus, angle of
    // the major axis).  Only DERIVED quantities (catalog columns, the
    // Python a1 / a2 / t properties) go through this; the fit itself
    // never does.  The tiny offset (1e-16 arcsec^4, far below any real
    // covariance) keeps the square root differentiable for an exactly
    // round source, and the same offset in the angle's derivative
    // makes it 0 instead of 0/0 there: the angle of a round source is
    // undefined, and so is its response.
    inline void eigen(
        math::qnumber& lp, math::qnumber& lm, math::qnumber& ang
    ) const {
        math::qnumber tr = this->mxx + this->myy;
        math::qnumber dx = this->mxx - this->myy;
        math::qnumber dy = 2.0 * this->mxy;
        math::qnumber diff = math::sqrt(dx * dx + dy * dy + 1.0e-16);
        lp = 0.5 * (tr + diff);
        lm = 0.5 * (tr - diff);
        const double denom = dx.v * dx.v + dy.v * dy.v + 1.0e-16;
        const double d1 = dx.v / denom;
        const double d2 = -dy.v / denom;
        ang = 0.5 * math::qnumber(
            std::atan2(dy.v, dx.v),
            d1 * dy.g1 + d2 * dx.g1,
            d1 * dy.g2 + d2 * dx.g2,
            d1 * dy.x1 + d2 * dx.x1,
            d1 * dy.x2 + d2 * dx.x2
        );
    };
public:
    bool force_size, force_center;
    math::qnumber F = math::qnumber(0.0);
    // intrinsic covariance, arcsec^2 (a round 0.2 arcsec source)
    math::qnumber mxx = math::qnumber(0.04);
    math::qnumber myy = math::qnumber(0.04);
    math::qnumber mxy = math::qnumber(0.0);
    math::qnumber x1;
    math::qnumber x2;
    // sigma^2 of the re-smoothing the source was fitted under (set by
    // GaussFit); get_shape reports the ellipticity at that scale
    double sigma2_shape = 0.0;

    NgmixGaussian(
        bool force_size=false,
        bool force_center=false
    ) :
        force_size(force_size),
        force_center(force_center){};

    // M from the semi-axes a1 (along angle t), a2 and the angle t
    inline void set_axes(
        const math::qnumber& a1, const math::qnumber& a2,
        const math::qnumber& t
    ) {
        math::qnumber c = math::cos(t);
        math::qnumber s = math::sin(t);
        math::qnumber m1 = a1 * a1;
        math::qnumber m2 = a2 * a2;
        this->mxx = m1 * c * c + m2 * s * s;
        this->myy = m1 * s * s + m2 * c * c;
        this->mxy = (m1 - m2) * c * s;
    };

    // M from its eigenvalues (major, minor, in arcsec^2) and the angle
    inline void set_eigen(
        const math::qnumber& lp, const math::qnumber& lm,
        const math::qnumber& t
    ) {
        math::qnumber c = math::cos(t);
        math::qnumber s = math::sin(t);
        this->mxx = lp * c * c + lm * s * s;
        this->myy = lp * s * s + lm * c * c;
        this->mxy = (lp - lm) * c * s;
    };

    // (a1, a2, t) derived from M.  A negative eigenvalue (allowed: the
    // smoothed covariance is what must stay positive) is reported as a
    // (near) zero semi-axis through a smooth floor; the floor sits at
    // 1e-6 arcsec^2 rather than 0 so that sqrt keeps a finite
    // derivative there.
    inline std::array<math::qnumber, 3> get_axes() const {
        math::qnumber lp, lm, ang;
        this->eigen(lp, lm, ang);
        return {
            math::sqrt(smooth_max(lp, 1.0e-6, 1.0e-6)),
            math::sqrt(smooth_max(lm, 1.0e-6, 1.0e-6)),
            ang
        };
    };

    inline StampBounds get_stamp_bounds(
        const geometry::cell& cell,
        int rr
    ) const {
        int i_cen = static_cast<int>(
            std::round(this->x1.v / cell.scale)
        ) - cell.xmin ;
        int j_cen = static_cast<int>(
            std::round(this->x2.v / cell.scale)
        ) - cell.ymin;

        // Clip about the centre; clamping i_max against i_min would shift the
        // window inward (not clip it) when the source sits near a cell edge.
        int i_min = std::max(i_cen - rr, 0);
        int i_max = std::min(i_cen + rr + 1, cell.nx);

        int j_min = std::max(j_cen - rr, 0);
        int j_max = std::min(j_cen + rr + 1, cell.ny);

        return {i_min, i_max, j_min, j_max, i_cen, j_cen, rr};
    };

    // Adaptive-radius variant: 6 sigma of the model size plus margin,
    // clamped to [24, 60] pixels.
    inline StampBounds get_stamp_bounds(
        const geometry::cell& cell
    ) const {
        const double size = std::sqrt(
            std::max(this->mxx.v + this->myy.v, 0.0)
        );
        int rr = static_cast<int>(
            std::max(std::min(size / cell.scale * 6 + 12, 60.0), 24.0)
        );
        return this->get_stamp_bounds(cell, rr);
    };

    inline modelKernelB
    prepare_modelB(double scale, double sigma_arcsec) const {
        double scale2 = one_over_two_pi * scale * scale;
        modelKernelB kernel;
        kernel.scale = scale;
        double sigma2 = sigma_arcsec * sigma_arcsec;
        math::qnumber cxx = this->mxx + sigma2;
        math::qnumber cyy = this->myy + sigma2;
        const math::qnumber& cxy = this->mxy;
        math::qnumber det_inv = 1.0 / (cxx * cyy - cxy * cxy);
        kernel.ixx = cyy * det_inv;
        kernel.iyy = cxx * det_inv;
        kernel.ixy = -1.0 * cxy * det_inv;
        kernel.f = math::pow(det_inv, 0.5) * scale2;
        return kernel;
    };

    inline modelKernelD
    prepare_modelD(double scale, double sigma_arcsec) const {
        double scale2 = one_over_two_pi * scale * scale;
        modelKernelD kernel;
        kernel.scale = scale;
        double sigma2 = sigma_arcsec * sigma_arcsec;
        math::qnumber cxx = this->mxx + sigma2;
        math::qnumber cyy = this->myy + sigma2;
        const math::qnumber& cxy = this->mxy;
        math::qnumber det_inv = 1.0 / (cxx * cyy - cxy * cxy);
        const math::qnumber ixx = cyy * det_inv;
        const math::qnumber iyy = cxx * det_inv;
        const math::qnumber ixy = -1.0 * cxy * det_inv;
        kernel.ixx = ixx;
        kernel.iyy = iyy;
        kernel.ixy = ixy;
        kernel.f = math::pow(det_inv, 0.5) * scale2;
        if (!this->force_size) {
            // d ln f / dM = -1/2 d ln det C / dM = -1/2 (C^{-1})_ab
            // (the off-diagonal counts twice)
            kernel.f_mxx = -0.5 * kernel.f * ixx;
            kernel.f_myy = -0.5 * kernel.f * iyy;
            kernel.f_mxy = -1.0 * kernel.f * ixy;
            // dI / dC_ab = -I E_ab I
            kernel.dI[0] = {
                -1.0 * ixx * ixx, -1.0 * ixx * ixy, -1.0 * ixy * ixy
            };
            kernel.dI[1] = {
                -1.0 * ixy * ixy, -1.0 * ixy * iyy, -1.0 * iyy * iyy
            };
            kernel.dI[2] = {
                -2.0 * ixx * ixy,
                -1.0 * (ixx * iyy + ixy * ixy),
                -2.0 * ixy * iyy
            };
        }
        return kernel;
    };

    inline math::qnumber get_r2(
        double x,
        double y,
        const modelKernelB & c
    ) const {
        math::qnumber xs = x - this->x1;
        math::qnumber ys = y - this->x2;
        return c.ixx * xs * xs + 2.0 * c.ixy * xs * ys + c.iyy * ys * ys;
    };

    inline math::lossNumber get_r2(
        double x,
        double y,
        const modelKernelD & c
    ) const {
        math::qnumber xs = x - this->x1;
        math::qnumber ys = y - this->x2;
        math::qnumber xx = xs * xs;
        math::qnumber xy = xs * ys;
        math::qnumber yy = ys * ys;

        math::lossNumber result;
        result.v = c.ixx * xx + 2.0 * c.ixy * xy + c.iyy * yy;
        if (!this->force_size) {
            result.v_mxx = c.dI[0][0] * xx + 2.0 * c.dI[0][1] * xy
                + c.dI[0][2] * yy;
            result.v_myy = c.dI[1][0] * xx + 2.0 * c.dI[1][1] * xy
                + c.dI[1][2] * yy;
            result.v_mxy = c.dI[2][0] * xx + 2.0 * c.dI[2][1] * xy
                + c.dI[2][2] * yy;
        }
        if (!this->force_center) {
            result.v_x1 = -2.0 * (c.ixx * xs + c.ixy * ys);
            result.v_x2 = -2.0 * (c.ixy * xs + c.iyy * ys);
        }
        return result;
    };

    inline math::qnumber get_model_from_r2(
        const math::qnumber& r2,
        const modelKernelB& c
    ) const {
        frDeriv fr = this->get_fr(r2);
        return this->F * fr.fr * c.f;
    };

    // The model at unit flux, m~ = m / F: what the flux pre-pass in
    // GaussFit::solve_flux sums against the data.
    inline math::qnumber get_unit_model_from_r2(
        const math::qnumber& r2,
        const modelKernelB& c
    ) const {
        frDeriv fr = this->get_fr(r2);
        return fr.fr * c.f;
    };

    inline math::lossNumber get_model_from_r2(
        const math::lossNumber& r2,
        const modelKernelD& c
    ) const {
        frDeriv fr = this->get_fr(r2.v);
        math::lossNumber res;
        res.v_F = fr.fr * c.f;
        res.v = this->F * res.v_F;

        fr.fr = fr.fr * this->F;
        fr.dfr = fr.dfr * this->F;

        math::qnumber f1 = fr.dfr * c.f;
        if (!this->force_size) {
            res.v_mxx = f1 * r2.v_mxx + fr.fr * c.f_mxx;
            res.v_myy = f1 * r2.v_myy + fr.fr * c.f_myy;
            res.v_mxy = f1 * r2.v_mxy + fr.fr * c.f_mxy;
        }
        if (!this->force_center) {
            res.v_x1 = f1 * r2.v_x1;
            res.v_x2 = f1 * r2.v_x2;
        }
        return res;
    };

    inline math::lossNumber get_model(
        double x, double y,
        const modelKernelD& c
    ) const {
        math::lossNumber r2 = this->get_r2(x, y, c);
        return get_model_from_r2(r2, c);
    };

    inline std::array<math::qnumber, 4> get_fpfs_moments(
        math::qnumber img_val,
        double x, double y,
        double rfac
    ) const {
        math::qnumber xs = x - this->x1;
        math::qnumber ys = y - this->x2;
        math::qnumber xx = xs * xs;
        math::qnumber yy = ys * ys;
        math::qnumber xy = xs * ys;
        math::qnumber model = math::exp((xx + yy) * rfac) * img_val;
        return {model, model * xx, model * yy, model * xy};
    };

    // chi2 of one pixel with its gradient and Gauss-Newton curvature
    // (diagonal and cross terms) in the free parameters
    inline math::lossNumber get_loss(
        const math::qnumber img_val,
        double variance_val,
        const math::lossNumber& r2,
        const modelKernelD & c
    ) const {
        math::lossNumber res;
        math::lossNumber th = this->get_model_from_r2(r2, c);
        math::qnumber residual = img_val - th.v;

        res.v = math::pow(residual, 2.0) / variance_val;
        res.v_in = res.v;
        double mul = 2.0 / variance_val;

        math::qnumber tmp = -1.0 * residual * mul;
        res.v_F = tmp * th.v_F;
        res.v_FF = math::pow(th.v_F, 2.0) * mul;
        if (!this->force_size) {
            res.v_Fmxx = th.v_F * th.v_mxx * mul;
            res.v_Fmyy = th.v_F * th.v_myy * mul;
            res.v_Fmxy = th.v_F * th.v_mxy * mul;
            res.v_mxx = tmp * th.v_mxx;
            res.v_myy = tmp * th.v_myy;
            res.v_mxy = tmp * th.v_mxy;
            res.v_mxxmxx = math::pow(th.v_mxx, 2.0) * mul;
            res.v_myymyy = math::pow(th.v_myy, 2.0) * mul;
            res.v_mxymxy = math::pow(th.v_mxy, 2.0) * mul;
            res.v_mxxmyy = th.v_mxx * th.v_myy * mul;
            res.v_mxxmxy = th.v_mxx * th.v_mxy * mul;
            res.v_myymxy = th.v_myy * th.v_mxy * mul;
        }
        if (!this->force_center) {
            res.v_Fx1 = th.v_F * th.v_x1 * mul;
            res.v_Fx2 = th.v_F * th.v_x2 * mul;
            res.v_x1 = tmp * th.v_x1;
            res.v_x2 = tmp * th.v_x2;
            res.v_x1x1 = math::pow(th.v_x1, 2.0) * mul;
            res.v_x2x2 = math::pow(th.v_x2, 2.0) * mul;
            res.v_x1x2 = th.v_x1 * th.v_x2 * mul;
            if (!this->force_size) {
                res.v_mxxx1 = th.v_mxx * th.v_x1 * mul;
                res.v_mxxx2 = th.v_mxx * th.v_x2 * mul;
                res.v_myyx1 = th.v_myy * th.v_x1 * mul;
                res.v_myyx2 = th.v_myy * th.v_x2 * mul;
                res.v_mxyx1 = th.v_mxy * th.v_x1 * mul;
                res.v_mxyx2 = th.v_mxy * th.v_x2 * mul;
            }
        }
        res.n_pix = 1.0;
        return res;
    };

    inline void
    add_to_cell(
        std::vector<math::qnumber> & data_model,
        const geometry::cell & cell,
        const modelKernelB & kernel
    ) const {
        const StampBounds bb = this->get_stamp_bounds(cell);
        for (int j = bb.j_min; (j < bb.j_max); ++j) {
            if (!cell.ymsk[j]) continue;
            int jj = j * cell.nx;
            for (int i = bb.i_min; (i < bb.i_max); ++i) {
                if (!cell.xmsk[i]) continue;
                if (bb.has_point(i, j)) {
                    math::qnumber r2 = this->get_r2(
                        cell.xvs[i], cell.yvs[j], kernel
                    );
                    data_model[jj + i] = (
                        data_model[jj + i] + this->get_model_from_r2(r2, kernel)
                    );
                }
            }
        }
        return;
    };

    // One epoch of the shape / centre update: a damped Gauss-Newton
    // step on all free parameters TOGETHER,
    //     (H fac + diag(w_prior + floor)) step = grad + prior,
    //     fac = 1 + lam + misfit * max_s(chi2_in / (n_in ref) - 1, 0),
    // H being the full curvature matrix (sizes and centre are coupled
    // through the normalisation and the quadratic form; a parameter-
    // by-parameter step converged only linearly and took tens of
    // epochs).  The flux is NOT stepped here: the model is linear in F,
    // so GaussFit::solve_flux sets it to its exact optimum for the
    // current shape before every loss evaluation (profiling F out).
    //
    // Damping.  ``lam`` is a DATA-INDEPENDENT schedule set by the
    // caller (large on the first epoch, shrinking afterwards).  The
    // misfit term is data dependent but a SMOOTH function of the loss,
    // never an accept / reject branch: a source whose model is far from
    // the pixels it covers -- a blend, a neighbour on top of it --
    // moves slowly, a well-fitted one takes an almost full Newton step.
    // chi2_in / n_in is the reduced chi2 over the model's own footprint
    // (a neighbour elsewhere in the window is not this source's
    // misfit), and ``ref`` is the reduced chi2 a perfect model reaches
    // on this cell, measured by the caller from the data's own noise
    // level: on a coadd with correlated noise the nominal variance
    // under-estimates the smoothed one by ~ 1.5, and referenced to 1 the
    // term fired on every source and slowed the whole cell to a crawl.
    // (The chi2 value itself used to sit in the denominator, ~ n_pix ~
    // 1e3 even for a perfect fit and comparable to the shape curvature
    // at moderate S/N, so every epoch took about half a step.)  ``floor``
    // is a constant added to the diagonal: it keeps the system positive
    // definite whatever the curvature and bounds the step of a
    // parameter with no curvature at all.  On top of that a smooth
    // trust radius caps the length of the centre step (arcsec) and of
    // the shape step (arcsec^2) per epoch, so that no gradient -- a
    // bright neighbour's, say -- can carry a source across the window
    // in one epoch.
    //
    // Convergence.  A step is a qnumber: its VALUE moves the parameter
    // and its four derivative slots move the parameter's shear /
    // position response.  Both converge at the same rate but from
    // different starting errors -- a source whose start is already at
    // the minimum in value can still carry the start's response, far
    // from the fit's -- so the fit is not done until the whole qnumber
    // has stopped moving.  The measure returned is
    //     c = grad . step.v  +  sum_slots step.s^T A step.s
    // (the predicted chi2 decrease, and the same per unit shear / unit
    // position shift), and the caller stops the source once
    // c < conv_tol.  The stop is a hard one, so the estimator jumps by
    // the size of the LAST step wherever the epoch count changes: at
    // most ~ sqrt(2 conv_tol / curv) in every slot, negligible at the
    // default tolerance.  A smooth gate on the value criterion was
    // tried instead and rejected: its derivative, (ds/dc)(dc/dg),
    // diverges as the value error goes to zero while the response
    // slots have not, and inside the ramp the estimator's derivative
    // took huge, rapidly varying values.
    //
    // Should the factorisation fail (it cannot for floor > 0, the
    // Gauss-Newton matrix being positive semi-definite), the epoch
    // falls back to the decoupled per-parameter step.
    inline double
    update_model_params(
        const math::lossNumber& loss,
        const modelPrior& prior,
        double x1_det,
        double x2_det,
        double lam,
        double floor,
        double trust_shape,
        double trust_center,
        double misfit,
        double misfit_ref,
        double sigma2_guard
    ) {
        math::qnumber fac = math::qnumber(1.0 + lam);
        if (misfit > 0.0 && loss.n_pix > 0.0 && misfit_ref > 0.0) {
            fac = fac + misfit * smooth_max(
                loss.v_in * (1.0 / (loss.n_pix * misfit_ref)) - 1.0,
                0.0, 0.1
            );
        }
        // free parameters, in the order mxx, myy, mxy, x1, x2
        std::array<math::qnumber*, 5> par{};
        std::array<math::qnumber, 5> b{};
        std::array<std::array<math::qnumber, 5>, 5> A{};
        int n = 0;
        int i0 = -1, i1 = -1, i2 = -1, ix1 = -1, ix2 = -1;
        if (!this->force_size) {
            i0 = n; par[n] = &this->mxx;
            b[n] = loss.v_mxx + prior.w_a * this->mxx;
            A[n][n] = loss.v_mxxmxx * fac + prior.w_a + floor; ++n;
            i1 = n; par[n] = &this->myy;
            b[n] = loss.v_myy + prior.w_a * this->myy;
            A[n][n] = loss.v_myymyy * fac + prior.w_a + floor; ++n;
            i2 = n; par[n] = &this->mxy;
            b[n] = loss.v_mxy + prior.w_a * this->mxy;
            A[n][n] = loss.v_mxymxy * fac + prior.w_a + floor; ++n;
            A[i0][i1] = loss.v_mxxmyy * fac;
            A[i0][i2] = loss.v_mxxmxy * fac;
            A[i1][i2] = loss.v_myymxy * fac;
        }
        if (!this->force_center) {
            ix1 = n; par[n] = &this->x1;
            b[n] = loss.v_x1 + prior.w_x * (this->x1 - x1_det);
            A[n][n] = loss.v_x1x1 * fac + prior.w_x + floor; ++n;
            ix2 = n; par[n] = &this->x2;
            b[n] = loss.v_x2 + prior.w_x * (this->x2 - x2_det);
            A[n][n] = loss.v_x2x2 * fac + prior.w_x + floor; ++n;
            A[ix1][ix2] = loss.v_x1x2 * fac;
            if (!this->force_size) {
                A[i0][ix1] = loss.v_mxxx1 * fac;
                A[i0][ix2] = loss.v_mxxx2 * fac;
                A[i1][ix1] = loss.v_myyx1 * fac;
                A[i1][ix2] = loss.v_myyx2 * fac;
                A[i2][ix1] = loss.v_mxyx1 * fac;
                A[i2][ix2] = loss.v_mxyx2 * fac;
            }
        }
        if (n == 0) return 0.0;
        // The flux is profiled out (solve_flux), so the curvature that
        // governs the shape / centre step is that of the PROFILED chi2:
        // the Schur complement H - h_F h_F^T / H_FF, with h_F the
        // flux-shape cross terms.  Stepping with the fixed-flux
        // curvature instead is block coordinate descent between the
        // flux and the shape, which converges only linearly.  (The
        // flux prior enters H_FF exactly as in solve_flux.)
        {
            std::array<math::qnumber, 5> hF{};
            if (!this->force_size) {
                hF[i0] = loss.v_Fmxx; hF[i1] = loss.v_Fmyy; hF[i2] = loss.v_Fmxy;
            }
            if (!this->force_center) {
                hF[ix1] = loss.v_Fx1; hF[ix2] = loss.v_Fx2;
            }
            math::qnumber hFF = loss.v_FF + prior.w_F;
            if (hFF.v > 0.0) {
                math::qnumber inv = fac / hFF;
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j <= i; ++j) {
                        A[j][i] = A[j][i] - hF[i] * hF[j] * inv;
                    }
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) A[i][j] = A[j][i];
        }
        std::array<math::qnumber, 5> step{};
        if (!solve_spd_qnumber(n, A, b, step)) {
            for (int i = 0; i < n; ++i) {
                step[i] = b[i] / A[i][i];
            }
        }
        // smooth trust radii (the shape step in the Frobenius norm)
        if (!this->force_size && trust_shape > 0.0) {
            math::qnumber d2 = step[i0] * step[i0] + step[i1] * step[i1]
                + 2.0 * step[i2] * step[i2];
            math::qnumber s = trust_factor(d2, trust_shape);
            step[i0] = step[i0] * s;
            step[i1] = step[i1] * s;
            step[i2] = step[i2] * s;
        }
        if (!this->force_center && trust_center > 0.0) {
            math::qnumber d2 = step[ix1] * step[ix1] + step[ix2] * step[ix2];
            math::qnumber s = trust_factor(d2, trust_center);
            step[ix1] = step[ix1] * s;
            step[ix2] = step[ix2] * s;
        }
        // The smoothed covariance C = M + sigma^2 I must stay positive
        // definite.  The chi2 itself is a barrier (the model diverges
        // as det C -> 0) and the trust radius bounds each move, so a
        // step never reaches the boundary for any sensible source; the
        // check below is the last resort for a pathological one and
        // simply shortens ITS step (a hard branch, but one no real fit
        // takes: the whole window would have to pull the covariance
        // through the barrier).
        if (!this->force_size && sigma2_guard > 0.0) {
            double shrink = 1.0;
            for (int k = 0; k < 12; ++k) {
                const double cxx = this->mxx.v - shrink * step[i0].v + sigma2_guard;
                const double cyy = this->myy.v - shrink * step[i1].v + sigma2_guard;
                const double cxy = this->mxy.v - shrink * step[i2].v;
                if (cxx > 0.0 && cyy > 0.0 &&
                    cxx * cyy - cxy * cxy > 0.01 * sigma2_guard * sigma2_guard) {
                    break;
                }
                shrink *= 0.5;
            }
            if (shrink < 1.0) {
                step[i0] = step[i0] * shrink;
                step[i1] = step[i1] * shrink;
                step[i2] = step[i2] * shrink;
            }
        }
        double c = 0.0;
        for (int i = 0; i < n; ++i) {
            c += b[i].v * step[i].v;
            for (int j = 0; j < n; ++j) {
                c += A[i][j].v * (
                    step[i].g1 * step[j].g1 + step[i].g2 * step[j].g2 +
                    step[i].x1 * step[j].x1 + step[i].x2 * step[j].x2
                );
            }
            *par[i] = *par[i] - step[i];
        }
        return c;
    };

    // Ellipticity of the model at the re-smoothing scale,
    //     (e1, e2) = ((cxx - cyy), 2 cxy) / (cxx + cyy),  C = M + sigma^2 I,
    // i.e. of the Gaussian that is actually compared with the
    // (deconvolved, re-smoothed) pixels.  C is positive definite, so
    // |e| < 1 always.  The INTRINSIC ellipticity (mxx - myy) / T is not
    // usable as a per-object estimator: for an unresolved source the
    // intrinsic covariance is zero within noise and can be negative,
    // so that ratio is unbounded (|e| ~ 30 and |R| > 5 for a third of
    // the sources on a DP1 patch).  The smoothing dilutes the shear
    // signal by ~ T / (T + 2 sigma^2), which the propagated response
    // calibrates exactly, and an unresolved source gets a small
    // response instead of a wild shape.  With sigma2_shape = 0 (a
    // model that was never fitted) this is the intrinsic ellipticity
    // with a small positive floor on T.
    inline std::array<math::qnumber, 2>
    get_shape() const {
        math::qnumber tr = this->mxx + this->myy + 2.0 * this->sigma2_shape;
        if (this->sigma2_shape <= 0.0) {
            // floor at 2 x (0.05 arcsec)^2, 1e-5 arcsec^2 hand-over
            tr = smooth_max(tr, 0.005, 1.0e-5);
        }
        math::qnumber e1 = (this->mxx - this->myy) / tr;
        math::qnumber e2 = 2.0 * this->mxy / tr;
        return {e1, e2};
    }

    inline math::qnumber
    get_flux_stamp(
        int nx,
        int ny,
        double scale,
        double sigma_arcsec
    ) const {
        int x_stamp = static_cast<int>(
            std::round(this->x1.v / scale)
        );
        int y_stamp = static_cast<int>(
            std::round(this->x2.v / scale)
        );
        modelKernelD c = this->prepare_modelD(scale, sigma_arcsec);
        int nx2 = nx / 2;
        int ny2 = ny / 2;
        math::qnumber flux;
        for (int j = 0; j < ny; ++j) {
            double y = (j - ny2 + y_stamp) * scale;
            for (int i = 0; i < nx; ++i) {
                double x = (i - nx2 + x_stamp) * scale;
                math::lossNumber r2 = this->get_r2(x, y, c);
                if (r2.v.v < apod_r2_hi) {
                    flux = flux + this->get_model_from_r2(r2, c).v;
                }
            }
        }
        return flux;
    }

    inline py::array_t<double>
    get_image_stamp(
        int nx,
        int ny,
        double scale,
        double sigma_arcsec
    ) const {
        int x_stamp = static_cast<int>(
            std::round(this->x1.v / scale)
        );
        int y_stamp = static_cast<int>(
            std::round(this->x2.v / scale)
        );
        modelKernelD c = this->prepare_modelD(scale, sigma_arcsec);
        auto result = py::array_t<double>({3, ny, nx});
        auto r = result.mutable_unchecked<3>();
        int nx2 = nx / 2;
        int ny2 = ny / 2;
        for (int j = 0; j < ny; ++j) {
            double y = (j - ny2 + y_stamp) * scale;
            for (int i = 0; i < nx; ++i) {
                double x = (i - nx2 + x_stamp) * scale;
                math::lossNumber r2 = this->get_r2(x, y, c);
                if (r2.v.v < apod_r2_hi) {
                    math::qnumber tn = this->get_model_from_r2(r2, c).v;
                    r(0, j, i) = tn.v;
                    r(1, j, i) = tn.g1;
                    r(2, j, i) = tn.g2;
                } else {
                    r(0, j, i) = 0.0;
                    r(1, j, i) = 0.0;
                    r(2, j, i) = 0.0;
                }
            }
        }
        return result;
    }

    inline void
    shift_reference(double dx1, double dx2) {
        // Signed reference shift of every fitted quantity; sign
        // conventions as in math::qnumber::shift_reference (positive =
        // reference to the cell center, i.e. centralize; negative =
        // back to the detection peak, i.e. decentralize).
        this->F.shift_reference(dx1, dx2);
        this->mxx.shift_reference(dx1, dx2);
        this->myy.shift_reference(dx1, dx2);
        this->mxy.shift_reference(dx1, dx2);
        this->x1.shift_reference(dx1, dx2);
        this->x2.shift_reference(dx1, dx2);
    };

    inline void
    centralize(double dx1, double dx2) {
        this->shift_reference(dx1, dx2);
    };

    inline void
    decentralize(double dx1, double dx2) {
        this->shift_reference(-dx1, -dx2);
    };

    // NOT virtual: nothing derives from NgmixGaussian, and a virtual
    // destructor would add a vptr and make galNumber (which embeds this
    // class) non-trivially-copyable -- every catalog copy would then be a
    // member-wise copy instead of a memcpy.
    ~NgmixGaussian() = default;
};


} // ngmix
} // anacal
#endif // ANACAL_NGMIX_RMODEL_H
