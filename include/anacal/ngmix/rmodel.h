#ifndef ANACAL_NGMIX_RMODEL_H
#define ANACAL_NGMIX_RMODEL_H

#include "../stdafx.h"

namespace anacal {
namespace ngmix {


struct frDeriv {
    // f(r) and its derivatives
    math::qnumber fr, dfr, ddfr;

    frDeriv() = default;

    frDeriv(
        math::qnumber fr, math::qnumber dfr, math::qnumber ddfr
    )
        : fr(fr), dfr(dfr), ddfr(ddfr) {}
};

struct lossNumber {
    // value with derivatives to Gaussian model parameters
    math::qnumber v, v_A, v_t, v_e1, v_e2, v_x1, v_x2;
    math::qnumber v_AA, v_tt, v_e1e1, v_e2e2, v_x1x1, v_x2x2;

    lossNumber() = default;

    lossNumber(
        math::qnumber v,
        math::qnumber v_A,
        math::qnumber v_t,
        math::qnumber v_e1,
        math::qnumber v_e2,
        math::qnumber v_x1,
        math::qnumber v_x2,
        math::qnumber v_AA,
        math::qnumber v_tt,
        math::qnumber v_e1e1,
        math::qnumber v_e2e2,
        math::qnumber v_x1x1,
        math::qnumber v_x2x2
    ) : v(v), v_A(v_A), v_t(v_t), v_e1(v_e1), v_e2(v_e2),
        v_x1(v_x1), v_x2(v_x2),
        v_AA(v_AA), v_tt(v_tt), v_e1e1(v_e1e1), v_e2e2(v_e2e2),
        v_x1x1(v_x1x1), v_x2x2(v_x2x2) {}

    // Define addition for lossNumber + lossNumber
    lossNumber operator+(const lossNumber& other) const {
        return lossNumber(
            this->v + other.v,
            this->v_A + other.v_A,
            this->v_t + other.v_t,
            this->v_e1 + other.v_e1,
            this->v_e2 + other.v_e2,
            this->v_x1 + other.v_x1,
            this->v_x2 + other.v_x2,
            this->v_AA + other.v_AA,
            this->v_tt + other.v_tt,
            this->v_e1e1 + other.v_e1e1,
            this->v_e2e2 + other.v_e2e2,
            this->v_x1x1 + other.v_x1x1,
            this->v_x2x2 + other.v_x2x2
        );
    }

    // Define subtraction for lossNumber - lossNumber
    lossNumber operator-(const lossNumber& other) const {
        return lossNumber(
            this->v - other.v,
            this->v_A - other.v_A,
            this->v_t - other.v_t,
            this->v_e1 - other.v_e1,
            this->v_e2 - other.v_e2,
            this->v_x1 - other.v_x1,
            this->v_x2 - other.v_x2,
            this->v_AA - other.v_AA,
            this->v_tt - other.v_tt,
            this->v_e1e1 - other.v_e1e1,
            this->v_e2e2 - other.v_e2e2,
            this->v_x1x1 - other.v_x1x1,
            this->v_x2x2 - other.v_x2x2
        );
    }

    // Define unary negation for -lossNumber
    lossNumber operator-() const {
        return lossNumber(
            -this->v,
            -this->v_A,
            -this->v_t,
            -this->v_e1,
            -this->v_e2,
            -this->v_x1,
            -this->v_x2,
            -this->v_AA,
            -this->v_tt,
            -this->v_e1e1,
            -this->v_e2e2,
            -this->v_x1x1,
            -this->v_x2x2
        );
    }
};

struct modelNumber {
    // value with derivatives to Gaussian model parameters
    math::qnumber A = {1.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber t = {-1.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber e1, e2, x1, x2;

    modelNumber() = default;

    modelNumber(
        math::qnumber A,
        math::qnumber t,
        math::qnumber e1,
        math::qnumber e2,
        math::qnumber x1,
        math::qnumber x2
    ) : A(A), t(t), e1(e1), e2(e2), x1(x1), x2(x2) {}
};

struct galNumber {
    // value with derivatives to Gaussian model parameters
    modelNumber params;
    math::qnumber wdet;
    lossNumber loss;

    galNumber() = default;

    galNumber(
        modelNumber params,
        math::qnumber wdet,
        lossNumber loss
    ) : params(params), wdet(wdet), loss(loss) {}
};


class NgmixModel {
private:
    math::qnumber v_p0, v_p1, v_p2;
    math::qnumber t_p0, t_p1, t_p2;
    math::qnumber e1_p0, e1_p1, e1_p2, e2_p0, e2_p1, e2_p2;
    math::qnumber x1_p1, x1_p2, x2_p1, x2_p2;
    math::qnumber tt_p0, tt_p1, tt_p2;
    math::qnumber e1e1_p0, e1e1_p1, e1e1_p2, e2e2_p0, e2e2_p1, e2e2_p2;
    math::qnumber x1x1, x2x2;

    virtual frDeriv get_fr(
        math::qnumber r2
    ) const {
        return frDeriv(r2, r2, r2);
    };

public:
    double sigma = 1.0;
    math::qnumber A, rho, e1, e2, x1, x2, t;   // parameters
    NgmixModel() {};
    math::qnumber det;

    void set_params(
        modelNumber params
    ) {
        this->A = params.A;
        this->t = params.t;
        this->rho = math::exp(this->t);
        this->e1 = params.e1;
        this->e2 = params.e2;
        this->x1 = params.x1;
        this->x2 = params.x2;
    };

    modelNumber get_params() {
        modelNumber params(
            this->A,
            this->t,
            this->e1,
            this->e2,
            this->x1,
            this->x2
        );
        return params;
    };

    void prepare_grad() {
        double sigma2 = this->sigma * this->sigma;
        math::qnumber rs2 = math::pow(this->rho, 2) + sigma2;
        math::qnumber ee = math::pow(this->e1, 2) + math::pow(this->e2, 2);
        math::qnumber det = math::pow(rs2, 2) - ee * math::pow(this->rho, 4);
        this->det = det;
        math::qnumber det_inv = 1.0 / det;
        math::qnumber det_inv2 = math::pow(det_inv, 2.0);
        math::qnumber det_inv3 = det_inv2 * det_inv;
        math::qnumber rho2 = math::pow(this->rho, 2.0);
        math::qnumber rho4 = rho2 * rho2;
        math::qnumber rho6 = rho2 * rho4;
        math::qnumber tmp1 = rs2 - rho2 * ee;
        math::qnumber e1e1 = e1 * e1;
        math::qnumber e1e2 = e1 * e2;
        math::qnumber e2e2 = e2 * e2;

        this->v_p0 = rs2 * det_inv;
        this->v_p1 = -e1 * rho2 * det_inv;
        this->v_p2 = -e2 * rho2 * det_inv;

        this->t_p0 = 2.0 * rho * (
            det_inv - 2.0 * rs2 * tmp1 * det_inv2
        ) * this->rho;
        this->t_p1 = 2.0 * e1 * rho * (
            -det_inv + 2.0 * rho2 * tmp1 * det_inv2
        ) * this->rho;
        this->t_p2 = 2.0 * e2 * rho * (
            -det_inv + 2.0 * rho2 * tmp1 * det_inv2
        ) * this->rho;

        this->e1_p0 = 2.0 * e1 * rho4 * rs2 * det_inv2;
        this->e1_p1 = rho2 * (-det_inv - 2.0 * e1e1 * rho4 * det_inv2);
        this->e1_p2 = -2.0 * e1e2 * rho4 * rho2 * det_inv2;

        this->e2_p0 = 2.0 * e2 * rho4 * rs2 * det_inv2;
        this->e2_p1 = -2.0 * e1e2 * rho6 * det_inv2;
        this->e2_p2 = rho2 * (-det_inv - 2.0 * e2e2 * rho4 * det_inv2);

        this->x1_p1 = -2.0 * (rs2 - e1 * rho2) * det_inv;
        this->x1_p2 = 2.0 * (e2 * rho2) * det_inv;
        this->x2_p1 = 2.0 * (e2 * rho2) * det_inv;
        this->x2_p2 = -2.0 * (rs2 + e1 * rho2) * det_inv;


        this->e1e1_p0 = 2.0 * rho4 * rs2 * det_inv3 * (
            rs2 * rs2 + rho4 * (3.0 * e1e1 - e2e2)
        );
        this->e1e1_p1 = -2.0 * e1 * rho6 * det_inv3 * (
            3.0 * rs2 * rs2 + rho4 * (e1e1 - 3.0 * e2e2)
        );
        this->e1e1_p2 = -2.0 * e2 * rho6 * det_inv3 * (
            rs2 * rs2 + rho4 * (3.0 * e1e1 - e2e2)
        );
        this->e2e2_p0 = 2.0 * rho4 * rs2 * det_inv3 * (
            rs2 * rs2 + rho4 * (3.0 * e2e2 - e1e1)
        );
        this->e2e2_p1 = -2.0 * e1 * rho6 * det_inv3 * (
            rs2 * rs2 + rho4 * (3.0 * e2e2 - e1e1)
        );
        this->e2e2_p2 = -2.0 * e2 * rho6 * det_inv3 * (
            3.0 * rs2 * rs2 + rho4 * (e2e2 - 3.0 * e1e1)
        );
        this->x1x1 = 2.0 * (rs2 - e1 * rho2) * det_inv;
        this->x2x2 = 2.0 * (rs2 + e1 * rho2) * det_inv;

        math::qnumber tmp2 = (
            3.0 * rho2 + sigma2 - 3.0 * e1e1 * rho2 - 3.0 * e2e2 * rho2
        );
        math::qnumber tmp3 = 8.0 * rho2 * pow(tmp1, 2) - det * tmp2;

        this-> tt_p0 = -2.0 * (
            8.0 * rho2 * tmp1 * det_inv2
            - 2.0 * rs2 * tmp3 * det_inv3
            - det_inv
        ) * rho2 + this->t_p0;
        this-> tt_p1 = -2.0 * e1 * (
            2.0 * rho2 * tmp3 * det_inv3
            - 8.0 * rho2 * tmp1 * det_inv2
            + det_inv
        ) * rho2 + this->t_p1;
        this-> tt_p2 = -2.0 * e2 * (
            2.0 * rho2 * tmp3 * det_inv3
            - 8.0 * rho2 * tmp1 * det_inv2
            + det_inv
        ) * rho2 + this->t_p2;
    }

    lossNumber get_r2(
        double x,
        double y
    ) const {

        // Center Shifting
        math::qnumber xs = x - this->x1;
        math::qnumber ys = y - this->x2;

        lossNumber result;
        math::qnumber q0 = xs * xs + ys * ys;
        math::qnumber q1 = xs * xs - ys * ys;
        math::qnumber q2 = 2.0 * xs * ys;

        result.v = v_p0 * q0 + v_p1 * q1 + v_p2 * q2;
        // First-order derivatives
        result.v_t = t_p0 * q0 + t_p1 * q1 + t_p2 * q2;
        result.v_e1 = e1_p0 * q0 + e1_p1 * q1 + e1_p2 * q2;
        result.v_e2 = e2_p0 * q0 + e2_p1 * q1 + e2_p2 * q2;
        result.v_x1 = x1_p1 * xs + x1_p2 * ys;
        result.v_x2 = x2_p1 * xs + x2_p2 * ys;

        // Second-order derivatives
        result.v_tt = tt_p0 * q0 + tt_p1 * q1 + tt_p2 * q2;
        result.v_e1e1 = e1e1_p0 * q0 + e1e1_p1 * q1 + e1e1_p2 * q2;
        result.v_e2e2 = e2e2_p0 * q0 + e2e2_p1 * q1 + e2e2_p2 * q2;
        result.v_x1x1 = x1x1;
        result.v_x2x2 = x2x2;
        return result;
    };

    lossNumber model(
        double x, double y
    ) const {

        lossNumber r2 = this->get_r2(x, y);
        frDeriv fr =  this->get_fr(r2.v);
        lossNumber res;
        res.v = fr.fr;

        // First-order derivatives
        res.v_A = fr.fr / this->A;
        res.v_t = fr.dfr * r2.v_t;
        res.v_e1 = fr.dfr * r2.v_e1;
        res.v_e2 = fr.dfr * r2.v_e2;
        res.v_x1 = fr.dfr * r2.v_x1;
        res.v_x2 = fr.dfr * r2.v_x2;

        // Second-order derivatives
        res.v_tt = fr.ddfr * r2.v_t * r2.v_t + fr.dfr * r2.v_tt;
        res.v_e1e1 = fr.ddfr * r2.v_e1 * r2.v_e1 + fr.dfr * r2.v_e1e1;
        res.v_e2e2 = fr.ddfr * r2.v_e2 * r2.v_e2 + fr.dfr * r2.v_e2e2;
        res.v_x1x1 = fr.ddfr * r2.v_x1 * r2.v_x1 + fr.dfr * r2.v_x1x1;
        res.v_x2x2 = fr.ddfr * r2.v_x2 * r2.v_x2 + fr.dfr * r2.v_x2x2;
        return res;
    };

    lossNumber loss(
        math::qnumber img_val,
        double variance_val,
        double x, double y
    ) const {
        lossNumber theory_val = this->model(x, y);
        math::qnumber residual = img_val - theory_val.v;

        lossNumber res;

        res.v = math::pow(residual, 2.0) / variance_val;
        double mul = 2.0 / variance_val;

        // First-order derivatives
        math::qnumber tmp = -1.0 * residual * mul;
        res.v_A =  tmp * theory_val.v_A ;
        res.v_t = tmp * theory_val.v_t;
        res.v_e1 = tmp * theory_val.v_e1;
        res.v_e2 = tmp * theory_val.v_e2;
        res.v_x1 = tmp * theory_val.v_x1;
        res.v_x2 = tmp * theory_val.v_x2;

        // Second-order derivatives
        res.v_AA = (
            (theory_val.v_A * theory_val.v_A) * mul
            + tmp * theory_val.v_AA
        );
        res.v_tt = (
            (theory_val.v_t * theory_val.v_t) * mul
            + tmp * theory_val.v_tt
        );
        res.v_e1e1 = (
            (theory_val.v_e1 * theory_val.v_e1) * mul
            + tmp * theory_val.v_e1e1
        );
        res.v_e2e2 = (
            (theory_val.v_e2 * theory_val.v_e2) * mul
            + tmp * theory_val.v_e2e2
        );
        res.v_x1x1 = (
            (theory_val.v_x1 * theory_val.v_x1) * mul
            + tmp * theory_val.v_x1x1
        );
        res.v_x2x2 = (
            (theory_val.v_x2 * theory_val.v_x2) * mul
            + tmp * theory_val.v_x2x2
        );
        return res;
    };

    math::qnumber
    get_flux_stamp(
        int nx,
        int ny,
        double scale
    ) const {
        int nx2 = nx / 2;
        int ny2 = ny / 2;
        math::qnumber flux;
        for (int j = 0; j < ny; ++j) {
            double y = (j - ny2) * scale;
            for (int i = 0; i < nx; ++i) {
                double x = (i - nx2) * scale;
                flux = flux + this->model(x, y).v;
            }
        }
        flux = flux * (scale * scale);
        return flux;
    }

    py::array_t<double>
    get_image_stamp(
        int nx,
        int ny,
        double scale
    ) const {
        auto result = py::array_t<double>({ny, nx});
        auto r = result.mutable_unchecked<2>();
        int nx2 = nx / 2;
        int ny2 = ny / 2;
        math::qnumber flux;
        for (int j = 0; j < ny; ++j) {
            double y = (j - ny2) * scale;
            for (int i = 0; i < nx; ++i) {
                double x = (i - nx2) * scale;
                r(j, i) = this->model(x, y).v.v;
            }
        }
        return result;
    }

    virtual ~NgmixModel() = default;
};


/// NgmixGaussian Function
class NgmixGaussian : public NgmixModel {
private:
    frDeriv get_fr(
        math::qnumber r2
    ) const {
        math::qnumber fr = this->A * math::exp(r2 * (-0.5));
        math::qnumber dfr = fr * (-0.5);
        math::qnumber ddfr = dfr * (-0.5);
        return frDeriv(fr, dfr, ddfr);
    };
public:
    // NgmixGaussian Profile
    NgmixGaussian(double sigma_arcsec) {
        this->sigma = sigma_arcsec;
    };
};

}
}
#endif // ANACAL_NGMIX_RMODEL_H
