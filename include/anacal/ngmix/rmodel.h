#ifndef ANACAL_NGMIX_RMODEL_H
#define ANACAL_NGMIX_RMODEL_H

#include "../stdafx.h"

namespace anacal {
namespace ngmix {

inline constexpr double one_over_two_pi = 0.5 / M_PI;

struct modelPrior {
    math::qnumber w_F, w_a, w_t, w_x;

    modelPrior() = default;

    inline void set_sigma_F(math::qnumber sigma_F){
        this->w_F = 2.0 / math::pow(sigma_F, 2.0);
    }; // loss is chi2

    inline void set_sigma_a(math::qnumber sigma_a){
        this->w_a = 2.0 / math::pow(sigma_a, 2.0);
    }; // loss is chi2

    inline void set_sigma_t(math::qnumber sigma_t){
        this->w_t = 2.0 / math::pow(sigma_t, 2.0);
    };
    inline void set_sigma_x(math::qnumber sigma_x){
        this->w_x = 2.0 / math::pow(sigma_x, 2.0);
    };
};

struct modelKernel {
    math::qnumber v_p0, v_p1, v_p2;
    math::qnumber a1_p0, a1_p1, a1_p2;
    math::qnumber a2_p0, a2_p1, a2_p2;
    math::qnumber t_p0, t_p1, t_p2;
    math::qnumber x1_p1, x1_p2, x2_p1, x2_p2;
    math::qnumber f, f_a1, f_a2;
    double scale;

    modelKernel() = default;
};

struct frDeriv {
    // f(r) and its derivatives
    math::qnumber fr, dfr, ddfr;

    frDeriv() = default;

    frDeriv(
        math::qnumber fr, math::qnumber dfr, math::qnumber ddfr
    )
        : fr(fr), dfr(dfr), ddfr(ddfr) {}
};

class NgmixGaussian {
private:
    frDeriv get_fr(
        math::qnumber r2
    ) const {
        math::qnumber fr = math::exp(r2 * (-0.5));
        math::qnumber dfr = fr * (-0.5);
        /* math::qnumber ddfr = dfr * (-0.5); */
        math::qnumber ddfr;
        return frDeriv(fr, dfr, ddfr);
    };
public:
    bool force_size, force_center;
    math::qnumber F = math::qnumber(1.0);
    math::qnumber t = math::qnumber(0.0);
    math::qnumber a1 = math::qnumber(0.15);
    math::qnumber a2 = math::qnumber(0.15);
    math::qnumber x1, x2;   // parameters
    NgmixGaussian(
        bool force_size=false,
        bool force_center=false
    ) :
        force_size(force_size),
        force_center(force_center){};

    inline modelKernel
    prepare_model(double scale, double sigma_arcsec) const {
        double scale2 = one_over_two_pi * scale * scale;
        modelKernel kernel;
        kernel.scale = scale;
        double sigma2 = sigma_arcsec * sigma_arcsec;
        math::qnumber base1 = math::pow(this->a1, 2) + sigma2;
        math::qnumber base2 = math::pow(this->a2, 2) + sigma2;
        math::qnumber det_inv = 1.0 / base1 / base2;
        math::qnumber tx2 = 2.0 * this->t;
        math::qnumber cos = math::cos(tx2);
        math::qnumber sin = math::sin(tx2);
        math::qnumber m0 = (base1 + base2) * det_inv;
        math::qnumber m1 = (base1 - base2) * cos * det_inv;
        math::qnumber m2 = (base1 - base2) * sin * det_inv;

        kernel.x1_p1 = m1 - m0;
        kernel.x1_p2 = m2;

        kernel.x2_p1 = m2;
        kernel.x2_p2 = -1.0 * (m1 + m0);

        kernel.t_p1 = m2;
        kernel.t_p2 = -1.0 * m1;

        kernel.v_p0 = 0.5 * m0;
        kernel.v_p1 = -0.5 * m1;
        kernel.v_p2 = -0.5 * m2;

        {
            math::qnumber det_inv2 = math::pow(det_inv, 2.0);
            kernel.a1_p0 = -1.0 * det_inv2 * this->a1 * base2 * base2;
            kernel.a1_p1 = kernel.a1_p0 * cos;
            kernel.a1_p2 = kernel.a1_p0 * sin;

            kernel.a2_p0 = -1.0 * det_inv2 * this->a2 * base1 * base1;
            kernel.a2_p1 = -1.0 * kernel.a2_p0 * cos;
            kernel.a2_p2 = -1.0 * kernel.a2_p0 * sin;
        }

        {
            math::qnumber det_inv0_5 = math::pow(det_inv, 0.5);
            math::qnumber det_inv1_5 = det_inv0_5 * det_inv;
            kernel.f =  det_inv0_5 * scale2;
            kernel.f_a1 = det_inv1_5 * scale2 * -1.0 * this->a1 * base2;
            kernel.f_a2 = det_inv1_5 * scale2 * -1.0 * this->a2 * base1;
            // math::qnumber tmp = (2.0 * math::pow(this->a1, 2) - sigma2);
            // kernel.f_a1a1 = kernel.f * tmp / math::pow(base1, 2);
            // tmp = (2.0 * math::pow(this->a2, 2) - sigma2);
            // kernel.f_a2a2 = kernel.f * tmp / math::pow(base2, 2);
        }
        return kernel;
    };

    inline math::lossNumber get_r2(
        double x,
        double y,
        const modelKernel & c
    ) const {

        // Center Shifting
        math::qnumber xs = x - this->x1;
        math::qnumber ys = y - this->x2;

        math::lossNumber result;
        math::qnumber q0 = xs * xs + ys * ys;
        math::qnumber q1 = xs * xs - ys * ys;
        math::qnumber q2 = 2.0 * xs * ys;

        result.v = c.v_p0 * q0 + c.v_p1 * q1 + c.v_p2 * q2;
        // First-order derivatives
        if (!this->force_size) {
            result.v_t = c.t_p0 * q0 + c.t_p1 * q1 + c.t_p2 * q2;
            result.v_a1 = c.a1_p0 * q0 + c.a1_p1 * q1 + c.a1_p2 * q2;
            result.v_a2 = c.a2_p0 * q0 + c.a2_p1 * q1 + c.a2_p2 * q2;
        }
        if (!this->force_center) {
            result.v_x1 = c.x1_p1 * xs + c.x1_p2 * ys;
            result.v_x2 = c.x2_p1 * xs + c.x2_p2 * ys;
        }
        return result;
    };

    inline math::lossNumber get_model(
        double x, double y,
        const modelKernel& c
    ) const {

        math::lossNumber r2 = this->get_r2(x, y, c);
        frDeriv fr =  this->get_fr(r2.v);

        math::lossNumber res;
        res.v = this->F * fr.fr * c.f;
        res.v_F = fr.fr * c.f; // F is special

        fr.fr = fr.fr * this->F;
        fr.dfr = fr.dfr * this->F;

        math::qnumber f1 = fr.dfr * c.f;
        // First-order derivatives
        if (!this->force_size) {
            res.v_t = f1 * r2.v_t;
            res.v_a1 = f1 * r2.v_a1 + fr.fr * c.f_a1;
            res.v_a2 = f1 * r2.v_a2 + fr.fr * c.f_a2;
        }
        if (!this->force_center) {
            res.v_x1 = f1 * r2.v_x1;
            res.v_x2 = f1 * r2.v_x2;
        }
        return res;
    };

    inline std::array<math::qnumber, 4> get_fpfs_moments(
        math::qnumber img_val,
        double x, double y,
        double sigma_arcsec
    ) const {
        double fac = 1.0 / sigma_arcsec / sigma_arcsec;
        math::qnumber xs = x - this->x1;
        math::qnumber ys = y - this->x2;
        math::qnumber q0 = xs * xs + ys * ys;
        math::qnumber q1 = xs * xs - ys * ys;
        math::qnumber q2 = 2.0 * xs * ys;
        math::qnumber r2 = q0 * fac;
        math::qnumber model = fac * 0.15915494 * math::exp(r2 * (-0.5)) * img_val;
        return {model, model * q0, model * q1, model * q2};
    };

    inline math::lossNumber get_loss(
        math::qnumber img_val,
        double variance_val,
        double x, double y,
        const modelKernel & c
    ) const {

        math::lossNumber theory_val = this->get_model(x, y, c);
        math::qnumber residual = img_val - theory_val.v;

        math::lossNumber res;

        res.v = math::pow(residual, 2.0) / variance_val;
        double mul = 2.0 / variance_val;

        // First-order derivatives
        math::qnumber tmp = -1.0 * residual * mul;
        res.v_F =  tmp * theory_val.v_F ;
        if (!this->force_size) {
            res.v_t = tmp * theory_val.v_t;
            res.v_a1 = tmp * theory_val.v_a1;
            res.v_a2 = tmp * theory_val.v_a2;
        }
        if (!this->force_center) {
            res.v_x1 = tmp * theory_val.v_x1;
            res.v_x2 = tmp * theory_val.v_x2;
        }

        // Second-order derivatives
        res.v_FF = (
            math::pow(theory_val.v_F, 2.0) * mul
        );
        if (!this->force_size) {
            res.v_tt = (
                math::pow(theory_val.v_t, 2.0) * mul
                /* + tmp * theory_val.v_tt */
            );
            res.v_a1a1 = (
                math::pow(theory_val.v_a1, 2.0) * mul
                /* + tmp * theory_val.v_a1a1 */
            );
            res.v_a2a2 = (
                math::pow(theory_val.v_a2, 2.0) * mul
                /* + tmp * theory_val.v_a2a2 */
            );
        }
        if (!this->force_center) {
            res.v_x1x1 = (
                math::pow(theory_val.v_x1, 2.0) * mul
                /* + tmp * theory_val.v_x1x1 */
            );
            res.v_x2x2 = (
                math::pow(theory_val.v_x2, 2.0) * mul
                /* + tmp * theory_val.v_x2x2 */
            );
        }
        return res;
    };

    inline void
    update_model_params(
        const math::lossNumber& loss,
        const modelPrior& prior,
        int epoch,
        double variance_val=1.0
    ) {
        math::qnumber damp = loss.v * 2.0 * std::exp(-epoch / 2.0);
        double ratio = 2.0 / variance_val;
        this->F = this->F - (
            (loss.v_F + prior.w_F * this->F) / (
                0.01 * ratio + loss.v_FF + prior.w_F
            )
        );
        if (!this->force_size) {
            this->t = this->t - (
                (loss.v_t + prior.w_t * this->t) / (
                    damp + (loss.v_tt + prior.w_t)
                )
            );
            this->a1 = this->a1 - (
                (loss.v_a1 + prior.w_a * this->a1) / (
                    damp + (loss.v_a1a1 + prior.w_a)
                )
            );
            this->a2 = this->a2 - (
                (loss.v_a2 + prior.w_a * this->a2) / (
                    damp + (loss.v_a2a2 + prior.w_a)
                )
            );
        }
        if (!this->force_center) {
            this->x1 = this->x1 - (
                loss.v_x1 / (
                    damp + (loss.v_x1x1 + prior.w_x)
                )
            );
            this->x2 = this->x2 - (
                loss.v_x2 / (
                    damp + (loss.v_x2x2 + prior.w_x)
                )
            );
        }
    };

    inline std::array<math::qnumber, 2>
    get_shape() const {
        math::qnumber base1 = math::pow(this->a1, 2);
        math::qnumber base2 = math::pow(this->a2, 2);
        math::qnumber base = (base1 - base2) / (base1 + base2) * 2.0;
        math::qnumber tx2 = 2.0 * this->t;
        math::qnumber e1 = base * math::cos(tx2);
        math::qnumber e2 = base * math::sin(tx2);
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
        modelKernel c = this->prepare_model(scale, sigma_arcsec);
        int nx2 = nx / 2;
        int ny2 = ny / 2;
        math::qnumber flux;
        for (int j = 0; j < ny; ++j) {
            double y = (j - ny2 + y_stamp) * scale;
            for (int i = 0; i < nx; ++i) {
                double x = (i - nx2 + x_stamp) * scale;
                flux = flux + this->get_model(x, y, c).v;
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
        modelKernel c = this->prepare_model(scale, sigma_arcsec);
        auto result = py::array_t<double>({3, ny, nx});
        auto r = result.mutable_unchecked<3>();
        int nx2 = nx / 2;
        int ny2 = ny / 2;
        math::qnumber flux;
        for (int j = 0; j < ny; ++j) {
            double y = (j - ny2 + y_stamp) * scale;
            for (int i = 0; i < nx; ++i) {
                double x = (i - nx2 + x_stamp) * scale;
                math::qnumber tn = this->get_model(x, y, c).v;
                r(0, j, i) = tn.v;
                r(1, j, i) = tn.g1;
                r(2, j, i) = tn.g2;
            }
        }
        return result;
    }

    inline NgmixGaussian
    decentralize(double dx1, double dx2) const {
        // (dx1, dx2) is the position of the source wrt center of block
        NgmixGaussian result= *this;
        result.F = this->F.decentralize(dx1, dx2);
        result.t = this->t.decentralize(dx1, dx2);
        result.a1 = this->a1.decentralize(dx1, dx2);
        result.a2 = this->a2.decentralize(dx1, dx2);
        result.x1 = this->x1.decentralize(dx1, dx2);
        result.x2 = this->x2.decentralize(dx1, dx2);
        return result;
    };

    inline NgmixGaussian
    centralize(double dx1, double dx2) const {
        // (dx1, dx2) is the position of the source wrt center of block
        NgmixGaussian result= *this;
        result.F = this->F.centralize(dx1, dx2);
        result.t = this->t.centralize(dx1, dx2);
        result.a1 = this->a1.centralize(dx1, dx2);
        result.a2 = this->a2.centralize(dx1, dx2);
        result.x1 = this->x1.centralize(dx1, dx2);
        result.x2 = this->x2.centralize(dx1, dx2);
        return result;
    };

    virtual ~NgmixGaussian() = default;
};


} // ngmix
} // anacal
#endif // ANACAL_NGMIX_RMODEL_H
