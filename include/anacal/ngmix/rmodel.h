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

struct modelNumber {
    // value with derivatives to Gaussian model parameters
    math::qnumber v, v_A, v_rho, v_e1, v_e2, v_x, v_y;
    math::qnumber v_AA, v_rhorho, v_e1e1, v_e2e2, v_xx, v_yy;

    modelNumber() = default;

    modelNumber(
        math::qnumber v,
        math::qnumber v_A,
        math::qnumber v_rho,
        math::qnumber v_e1,
        math::qnumber v_e2,
        math::qnumber v_x,
        math::qnumber v_y,
        math::qnumber v_AA,
        math::qnumber v_rhorho,
        math::qnumber v_e1e1,
        math::qnumber v_e2e2,
        math::qnumber v_xx,
        math::qnumber v_yy
    ) : v(v), v_A(v_A), v_rho(v_rho), v_e1(v_e1), v_e2(v_e2),
        v_x(v_x), v_y(v_y),
        v_AA(v_AA), v_rhorho(v_rhorho), v_e1e1(v_e1e1), v_e2e2(v_e2e2),
        v_xx(v_xx), v_yy(v_yy) {}

    // Define addition for modelNumber + modelNumber
    modelNumber operator+(const modelNumber& other) const {
        return modelNumber(
            this->v + other.v,
            this->v_A + other.v_A,
            this->v_rho + other.v_rho,
            this->v_e1 + other.v_e1,
            this->v_e2 + other.v_e2,
            this->v_x + other.v_x,
            this->v_y + other.v_y,
            this->v_AA + other.v_AA,
            this->v_rhorho + other.v_rhorho,
            this->v_e1e1 + other.v_e1e1,
            this->v_e2e2 + other.v_e2e2,
            this->v_xx + other.v_xx,
            this->v_yy + other.v_yy
        );
    }

    // Define subtraction for modelNumber - modelNumber
    modelNumber operator-(const modelNumber& other) const {
        return modelNumber(
            this->v - other.v,
            this->v_A - other.v_A,
            this->v_rho - other.v_rho,
            this->v_e1 - other.v_e1,
            this->v_e2 - other.v_e2,
            this->v_x - other.v_x,
            this->v_y - other.v_y,
            this->v_AA - other.v_AA,
            this->v_rhorho - other.v_rhorho,
            this->v_e1e1 - other.v_e1e1,
            this->v_e2e2 - other.v_e2e2,
            this->v_xx - other.v_xx,
            this->v_yy - other.v_yy
        );
    }

    // Define unary negation for -modelNumber
    modelNumber operator-() const {
        return modelNumber(
            -this->v,
            -this->v_A,
            -this->v_rho,
            -this->v_e1,
            -this->v_e2,
            -this->v_x,
            -this->v_y,
            -this->v_AA,
            -this->v_rhorho,
            -this->v_e1e1,
            -this->v_e2e2,
            -this->v_xx,
            -this->v_yy
        );
    }
};


class NgmixModel {
private:
    math::qnumber v_p0, v_p1, v_p2;
    math::qnumber rho_p0, rho_p1, rho_p2;
    math::qnumber e1_p0, e1_p1, e1_p2, e2_p0, e2_p1, e2_p2;
    math::qnumber x0_p1, x0_p2, y0_p1, y0_p2;
    math::qnumber rhorho_p0, rhorho_p1, rhorho_p2;
    math::qnumber e1e1_p0, e1e1_p1, e1e1_p2, e2e2_p0, e2e2_p1, e2e2_p2;
    math::qnumber x0x0, y0y0;

    virtual frDeriv get_fr(
        math::qnumber r2
    ) const {
        return frDeriv(r2, r2, r2);
    };

public:
    double sigma = 1.0;
    math::qnumber A, rho, e1, e2, x0, y0;   // parameters
    NgmixModel() {};
    math::qnumber det;

    void set_params(
        std::array<math::qnumber, 6> params
    ) {
        this->A = params[0];
        this->rho = params[1];
        this->e1 = params[2];
        this->e2 = params[3];
        this->x0 = params[4];
        this->y0 = params[5];
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

        this->rho_p0 = 2.0 * rho * (
            det_inv - 2.0 * rs2 * tmp1 * det_inv2
        );
        this->rho_p1 = 2.0 * e1 * rho * (
            -det_inv + 2.0 * rho2 * tmp1 * det_inv2
        );
        this->rho_p2 = 2.0 * e2 * rho * (
            -det_inv + 2.0 * rho2 * tmp1 * det_inv2
        );

        this->e1_p0 = 2.0 * e1 * rho4 * rs2 * det_inv2;
        this->e1_p1 = rho2 * (-det_inv - 2.0 * e1e1 * rho4 * det_inv2);
        this->e1_p2 = -2.0 * e1e2 * rho4 * rho2 * det_inv2;

        this->e2_p0 = 2.0 * e2 * rho4 * rs2 * det_inv2;
        this->e2_p1 = -2.0 * e1e2 * rho6 * det_inv2;
        this->e2_p2 = rho2 * (-det_inv - 2.0 * e2e2 * rho4 * det_inv2);

        this->x0_p1 = -2.0 * (rs2 - e1 * rho2) * det_inv;
        this->x0_p2 = 2.0 * (e2 * rho2) * det_inv;
        this->y0_p1 = 2.0 * (e2 * rho2) * det_inv;
        this->y0_p2 = -2.0 * (rs2 + e1 * rho2) * det_inv;


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
        this->x0x0 = 2.0 * (rs2 - e1 * rho2) * det_inv;
        this->y0y0 = 2.0 * (rs2 + e1 * rho2) * det_inv;

        math::qnumber tmp2 = (
            3.0 * rho2 + sigma2 - 3.0 * e1e1 * rho2 - 3.0 * e2e2 * rho2
        );
        math::qnumber tmp3 = 8.0 * rho2 * pow(tmp1, 2) - det * tmp2;

        this-> rhorho_p0 = -2.0 * (
            8.0 * rho2 * tmp1 * det_inv2
            - 2.0 * rs2 * tmp3 * det_inv3
            - det_inv
        );
        this-> rhorho_p1 = -2.0 * e1 * (
            2.0 * rho2 * tmp3 * det_inv3
            - 8.0 * rho2 * tmp1 * det_inv2
            + det_inv
        );
        this-> rhorho_p2 = -2.0 * e2 * (
            2.0 * rho2 * tmp3 * det_inv3
            - 8.0 * rho2 * tmp1 * det_inv2
            + det_inv
        );
    }

    modelNumber get_r2(
        double x,
        double y
    ) const {

        // Center Shifting
        math::qnumber xs = x - this->x0;
        math::qnumber ys = y - this->y0;

        modelNumber result;
        math::qnumber q0 = xs * xs + ys * ys;
        math::qnumber q1 = xs * xs - ys * ys;
        math::qnumber q2 = 2.0 * xs * ys;

        result.v = v_p0 * q0 + v_p1 * q1 + v_p2 * q2;
        // First-order derivatives
        result.v_rho = rho_p0 * q0 + rho_p1 * q1 + rho_p2 * q2;
        result.v_e1 = e1_p0 * q0 + e1_p1 * q1 + e1_p2 * q2;
        result.v_e2 = e2_p0 * q0 + e2_p1 * q1 + e2_p2 * q2;
        result.v_x = x0_p1 * xs + x0_p2 * ys;
        result.v_y = y0_p1 * xs + y0_p2 * ys;

        // Second-order derivatives
        result.v_rhorho = rhorho_p0 * q0 + rhorho_p1 * q1 + rhorho_p2 * q2;
        result.v_e1e1 = e1e1_p0 * q0 + e1e1_p1 * q1 + e1e1_p2 * q2;
        result.v_e2e2 = e2e2_p0 * q0 + e2e2_p1 * q1 + e2e2_p2 * q2;
        result.v_xx = x0x0;
        result.v_yy = y0y0;
        return result;
    };

    modelNumber model(
        double x, double y
    ) const {

        modelNumber r2 = this->get_r2(x, y);
        frDeriv fr =  this->get_fr(r2.v);
        modelNumber res;
        res.v = fr.fr;

        // First-order derivatives
        res.v_A = fr.fr / this->A;
        res.v_rho = fr.dfr * r2.v_rho;
        res.v_e1 = fr.dfr * r2.v_e1;
        res.v_e2 = fr.dfr * r2.v_e2;
        res.v_x = fr.dfr * r2.v_x;
        res.v_y = fr.dfr * r2.v_y;

        // Second-order derivatives
        res.v_rhorho = fr.ddfr * r2.v_rho * r2.v_rho + fr.dfr * r2.v_rhorho;
        res.v_e1e1 = fr.ddfr * r2.v_e1 * r2.v_e1 + fr.dfr * r2.v_e1e1;
        res.v_e2e2 = fr.ddfr * r2.v_e2 * r2.v_e2 + fr.dfr * r2.v_e2e2;
        res.v_xx = fr.ddfr * r2.v_x * r2.v_x + fr.dfr * r2.v_xx;
        res.v_yy = fr.ddfr * r2.v_y * r2.v_y + fr.dfr * r2.v_yy;
        return res;
    };

    modelNumber loss(
        math::qnumber img_val,
        double variance_val,
        double x, double y
    ) const {
        modelNumber theory_val = this->model(x, y);
        math::qnumber residual = img_val - theory_val.v;

        modelNumber res;

        res.v = residual * residual / variance_val;
        double mul = 2.0 / variance_val;

        // First-order derivatives
        math::qnumber tmp = -1.0 * residual * mul;
        res.v_A =  tmp * theory_val.v_A ;
        res.v_rho = tmp * theory_val.v_rho;
        res.v_e1 = tmp * theory_val.v_e1;
        res.v_e2 = tmp * theory_val.v_e2;
        res.v_x = tmp * theory_val.v_x;
        res.v_y = tmp * theory_val.v_y;

        // Second-order derivatives
        res.v_AA = (
            (theory_val.v_A * theory_val.v_A) * mul
            + tmp * theory_val.v_AA
        );
        res.v_rhorho = (
            (theory_val.v_rho * theory_val.v_rho) * mul
            + tmp * theory_val.v_rhorho
        );
        res.v_e1e1 = (
            (theory_val.v_e1 * theory_val.v_e1) * mul
            + tmp * theory_val.v_e1e1
        );
        res.v_e2e2 = (
            (theory_val.v_e2 * theory_val.v_e2) * mul
            + tmp * theory_val.v_e2e2
        );
        res.v_xx = (
            (theory_val.v_x * theory_val.v_x) * mul
            + tmp * theory_val.v_xx
        );
        res.v_yy = (
            (theory_val.v_y * theory_val.v_y) * mul
            + tmp * theory_val.v_yy
        );
        return res;
    };

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
