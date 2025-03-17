#ifndef ANACAL_NGMIX_RMODEL_H
#define ANACAL_NGMIX_RMODEL_H

#include "../stdafx.h"

namespace anacal {
namespace ngmix {

struct modelPrior {
    math::qnumber w_F, w_t, w_e, w_x;
    math::qnumber mu_F, mu_t;
    math::qnumber mu_e1, mu_e2;
    math::qnumber mu_x1, mu_x2;

    modelPrior() = default;

    inline void set_sigma_F(math::qnumber sigma_F){
        this->w_F = 2.0 / math::pow(sigma_F, 2.0);
    }; // loss is chi2

    inline void set_sigma_t(math::qnumber sigma_t){
        this->w_t = 2.0 / math::pow(sigma_t, 2.0);
    }; // loss is chi2

    inline void set_sigma_e(math::qnumber sigma_e){
        this->w_e = 2.0 / math::pow(sigma_e, 2.0);
    };
    inline void set_sigma_x(math::qnumber sigma_x){
        this->w_x = 2.0 / math::pow(sigma_x, 2.0);
    };
};

struct modelKernel {
    math::qnumber v_p0, v_p1, v_p2;
    math::qnumber t_p0, t_p1, t_p2;
    math::qnumber e1_p0, e1_p1, e1_p2;
    math::qnumber e2_p0, e2_p1, e2_p2;
    math::qnumber x1_p1, x1_p2, x2_p1, x2_p2;
    math::qnumber tt_p0, tt_p1, tt_p2;
    math::qnumber e1e1_p0, e1e1_p1, e1e1_p2;
    math::qnumber e2e2_p0, e2e2_p1, e2e2_p2;
    math::qnumber x1x1, x2x2;

    math::qnumber f, f_t, f_e1, f_e2;
    math::qnumber f_tt, f_e1e1, f_e2e2;

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
    bool force_size, force_shape, force_center;
    math::qnumber F = math::qnumber(1.0);
    math::qnumber t = math::qnumber(-1.0);
    math::qnumber e1, e2, x1, x2;   // parameters
    NgmixGaussian(
        bool force_size=false,
        bool force_shape=false,
        bool force_center=false
    ) :
        force_size(force_size), force_shape(force_shape),
        force_center(force_center){};

    inline modelKernel
    prepare_model(double scale, double sigma_arcsec) const {
        double one_over_pi = 1.0 / M_PI;
        modelKernel kernel;
        kernel.scale = scale;
        double sigma2 = sigma_arcsec * sigma_arcsec;
        math::qnumber rho = math::exp(this->t);
        math::qnumber rho2 = math::pow(rho, 2.0);
        math::qnumber rho4 = rho2 * rho2;
        math::qnumber rho6 = rho2 * rho4;
        math::qnumber rs2 = rho2 + sigma2;
        math::qnumber e1e1 = math::pow(e1, 2);
        math::qnumber e2e2 = math::pow(e2, 2);
        math::qnumber e1e2 = e1 * e2;
        math::qnumber ee = e1e1 + e2e2;
        math::qnumber det = math::pow(rs2, 2) - ee * rho4;
        math::qnumber det_inv = 1.0 / det;
        math::qnumber det_inv0_5 = math::pow(det_inv, 0.5);
        math::qnumber det_inv2 = math::pow(det_inv, 2.0);
        math::qnumber det_inv3 = det_inv2 * det_inv;

        math::qnumber tmp1 = rs2 - rho2 * ee;

        kernel.v_p0 = rs2 * det_inv;
        kernel.v_p1 = -e1 * rho2 * det_inv;
        kernel.v_p2 = -e2 * rho2 * det_inv;

        kernel.t_p0 = 2.0 * rho2 * (
            det_inv - 2.0 * rs2 * tmp1 * det_inv2
        );
        kernel.t_p1 = 2.0 * e1 * rho2 * (
            -det_inv + 2.0 * rho2 * tmp1 * det_inv2
        );
        kernel.t_p2 = 2.0 * e2 * rho2 * (
            -det_inv + 2.0 * rho2 * tmp1 * det_inv2
        );

        kernel.e1_p0 = 2.0 * e1 * rho4 * rs2 * det_inv2;
        kernel.e1_p1 = rho2 * (-det_inv - 2.0 * e1e1 * rho4 * det_inv2);
        kernel.e1_p2 = -2.0 * e1e2 * rho6 * det_inv2;

        kernel.e2_p0 = 2.0 * e2 * rho4 * rs2 * det_inv2;
        kernel.e2_p1 = -2.0 * e1e2 * rho6 * det_inv2;
        kernel.e2_p2 = rho2 * (-det_inv - 2.0 * e2e2 * rho4 * det_inv2);

        kernel.x1_p1 = -2.0 * (rs2 - e1 * rho2) * det_inv;
        kernel.x1_p2 = 2.0 * (e2 * rho2) * det_inv;
        kernel.x2_p1 = 2.0 * (e2 * rho2) * det_inv;
        kernel.x2_p2 = -2.0 * (rs2 + e1 * rho2) * det_inv;


        kernel.e1e1_p0 = 2.0 * rho4 * rs2 * det_inv3 * (
            rs2 * rs2 + rho4 * (3.0 * e1e1 - e2e2)
        );
        kernel.e1e1_p1 = -2.0 * e1 * rho6 * det_inv3 * (
            3.0 * rs2 * rs2 + rho4 * (e1e1 - 3.0 * e2e2)
        );
        kernel.e1e1_p2 = -2.0 * e2 * rho6 * det_inv3 * (
            rs2 * rs2 + rho4 * (3.0 * e1e1 - e2e2)
        );
        kernel.e2e2_p0 = 2.0 * rho4 * rs2 * det_inv3 * (
            rs2 * rs2 + rho4 * (3.0 * e2e2 - e1e1)
        );
        kernel.e2e2_p1 = -2.0 * e1 * rho6 * det_inv3 * (
            rs2 * rs2 + rho4 * (3.0 * e2e2 - e1e1)
        );
        kernel.e2e2_p2 = -2.0 * e2 * rho6 * det_inv3 * (
            3.0 * rs2 * rs2 + rho4 * (e2e2 - 3.0 * e1e1)
        );
        kernel.x1x1 = 2.0 * (rs2 - e1 * rho2) * det_inv;
        kernel.x2x2 = 2.0 * (rs2 + e1 * rho2) * det_inv;

        math::qnumber r2ee = rho2 * (1.0 - ee);
        math::qnumber tmp2 = 3.0 * r2ee + sigma2;
        math::qnumber tmp3 = 8.0 * rho2 * pow(tmp1, 2) - det * tmp2;

        kernel.tt_p0 = -2.0 * (
            8.0 * rho2 * tmp1 * det_inv2
            - 2.0 * rs2 * tmp3 * det_inv3
            - det_inv
        ) * rho2 + kernel.t_p0;
        kernel.tt_p1 = -2.0 * e1 * (
            2.0 * rho2 * tmp3 * det_inv3
            - 8.0 * rho2 * tmp1 * det_inv2
            + det_inv
        ) * rho2 + kernel.t_p1;
        kernel.tt_p2 = -2.0 * e2 * (
            2.0 * rho2 * tmp3 * det_inv3
            - 8.0 * rho2 * tmp1 * det_inv2
            + det_inv
        ) * rho2 + kernel.t_p2;

        math::qnumber det_inv1_5 = det_inv0_5 * det_inv;
        math::qnumber det_inv2_5 = det_inv0_5 * det_inv2;
        math::qnumber dtmp = one_over_pi * det_inv1_5;

        double scale2 = scale * scale;
        kernel.f = one_over_pi * 0.5 * det_inv0_5 * scale2;
        kernel.f_t = -dtmp * (
            sigma2 + r2ee
        ) * rho2 * scale2;
        kernel.f_e1 = 0.5 * dtmp * rho4 * e1 * scale2;
        kernel.f_e2 = 0.5 * dtmp * rho4 * e2 * scale2;

        kernel.f_tt = 0.25 * one_over_pi * (
            1.5 * det_inv2_5 * math::pow((4.0 * rho * (sigma2 + r2ee)), 2.0)
            - det_inv1_5 * (4.0 * sigma2 + 12.0 * r2ee)
        ) * rho2 * scale2 + kernel.f_t * scale2;

        kernel.f_e1e1 = 0.5 * one_over_pi * rho4 * det_inv1_5 * (
            3.0 * e1e1 * rho4 * det_inv + 1.0
        ) * scale2;
        kernel.f_e2e2 = 0.5 * one_over_pi * rho4 * det_inv1_5 * (
            3.0 * e2e2 * rho4 * det_inv + 1.0
        ) * scale2;
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
        }
        if (!this->force_shape) {
            result.v_e1 = c.e1_p0 * q0 + c.e1_p1 * q1 + c.e1_p2 * q2;
            result.v_e2 = c.e2_p0 * q0 + c.e2_p1 * q1 + c.e2_p2 * q2;
        }
        if (!this->force_center) {
            result.v_x1 = c.x1_p1 * xs + c.x1_p2 * ys;
            result.v_x2 = c.x2_p1 * xs + c.x2_p2 * ys;
        }

        // Second-order derivatives

        if (!this->force_size) {
            result.v_tt = c.tt_p0 * q0 + c.tt_p1 * q1 + c.tt_p2 * q2;
        }
        if (!this->force_shape) {
            result.v_e1e1 = c.e1e1_p0 * q0 + c.e1e1_p1 * q1 + c.e1e1_p2 * q2;
            result.v_e2e2 = c.e2e2_p0 * q0 + c.e2e2_p1 * q1 + c.e2e2_p2 * q2;
        }
        if (!this->force_center) {
            result.v_x1x1 = c.x1x1;
            result.v_x2x2 = c.x2x2;
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
            res.v_t = f1 * r2.v_t + fr.fr * c.f_t;
        }
        if (!this->force_shape) {
            res.v_e1 = f1 * r2.v_e1 + fr.fr * c.f_e1;
            res.v_e2 = f1 * r2.v_e2 + fr.fr * c.f_e2;
        }
        if (!this->force_center) {
            res.v_x1 = f1 * r2.v_x1;
            res.v_x2 = f1 * r2.v_x2;
        }

        if (!this->force_size) {
            // Second-order derivatives
            res.v_tt = (
                f1 * (r2.v_tt - 0.5 * math::pow(r2.v_t, 2.0))
                + fr.fr * (c.f_tt - r2.v_t * c.f_t)
            );
        }

        if (!this->force_shape) {
            res.v_e1e1 = (
                f1 * (r2.v_e1e1 - 0.5 * math::pow(r2.v_e1, 2.0))
                + fr.fr * (c.f_e1e1 - r2.v_e1 * c.f_e1)
            );
            res.v_e2e2 = (
                f1 * (r2.v_e2e2 - 0.5 * math::pow(r2.v_e2, 2.0))
                + fr.fr * (c.f_e2e2 - r2.v_e2 * c.f_e2)
            );
        }
        if (!this->force_center) {
            res.v_x1x1 = f1 * (r2.v_x1x1 - 0.5 * math::pow(r2.v_x1, 2.0));
            res.v_x2x2 = f1 * (r2.v_x2x2 - 0.5 * math::pow(r2.v_x2, 2.0));
        }
        return res;
    };

    inline std::array<math::qnumber, 4> get_fpfs_moments(
        math::qnumber img_val,
        double x, double y,
        double sigma_arcsec
    ) const {
        math::qnumber xs = x - this->x1;
        math::qnumber ys = y - this->x2;
        math::qnumber q0 = xs * xs + ys * ys;
        math::qnumber q1 = xs * xs - ys * ys;
        math::qnumber q2 = 2.0 * xs * ys;
        math::qnumber r2 = q0 / sigma_arcsec / sigma_arcsec;
        math::qnumber model = math::exp(r2 * (-0.5)) * img_val;
        return {model * q0, model * q1, model * q2, model};
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
        }
        if (!this->force_shape) {
            res.v_e1 = tmp * theory_val.v_e1;
            res.v_e2 = tmp * theory_val.v_e2;
        }

        if (!this->force_center) {
            res.v_x1 = tmp * theory_val.v_x1;
            res.v_x2 = tmp * theory_val.v_x2;
        }

        // Second-order derivatives
        res.v_FF = (
            math::pow(theory_val.v_F, 2.0) * mul
            /* + tmp * theory_val.v_FF */
        );
        if (!this->force_size) {
            res.v_tt = (
                math::pow(theory_val.v_t, 2.0) * mul
                /* + tmp * theory_val.v_tt */
            );
        }
        if (!this->force_shape) {
            res.v_e1e1 = (
                math::pow(theory_val.v_e1, 2.0) * mul
                /* + tmp * theory_val.v_e1e1 */
            );
            res.v_e2e2 = (
                math::pow(theory_val.v_e2, 2.0) * mul
                /* + tmp * theory_val.v_e2e2 */
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
            (loss.v_F + prior.w_F * (this->F - prior.mu_F)) / (
                0.01 * ratio + loss.v_FF + prior.w_F
            )
        );
        if (!this->force_size) {
            this->t = this->t - (
                (loss.v_t + prior.w_t * (this->t - prior.mu_t)) / (
                    damp + (loss.v_tt + prior.w_t)
                )
            );
        }
        if (!this->force_shape) {
            this->e1 = this->e1 - (
                (loss.v_e1 + prior.w_e * (this->e1 - prior.mu_e1)) / (
                    damp + (loss.v_e1e1 + prior.w_e)
                )
            );
            this->e2 = this->e2 - (
                (loss.v_e2 + prior.w_e * (this->e2 - prior.mu_e2)) / (
                    damp + (loss.v_e2e2 + prior.w_e)
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
        result.e1 = this->e1.decentralize(dx1, dx2);
        result.e2 = this->e2.decentralize(dx1, dx2);
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
        result.e1 = this->e1.centralize(dx1, dx2);
        result.e2 = this->e2.centralize(dx1, dx2);
        result.x1 = this->x1.centralize(dx1, dx2);
        result.x2 = this->x2.centralize(dx1, dx2);
        return result;
    };

    virtual ~NgmixGaussian() = default;
};


} // ngmix
} // anacal
#endif // ANACAL_NGMIX_RMODEL_H
