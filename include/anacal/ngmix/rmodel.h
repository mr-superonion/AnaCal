#ifndef ANACAL_NGMIX_RMODEL_H
#define ANACAL_NGMIX_RMODEL_H

#include "../stdafx.h"

namespace anacal {
namespace ngmix {


struct frDeriv {
    // f(r) and its derivatives
    math::qnumber fr = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber dfr = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber ddfr = {0.0, 0.0, 0.0, 0.0, 0.0};

    frDeriv(
        math::qnumber fr, math::qnumber dfr, math::qnumber ddfr
    )
        : fr(fr), dfr(dfr), ddfr(ddfr) {}
};

struct modelNumber {
    // value with derivatives to Gaussian model parameters
    math::qnumber v = {0.0, 0.0, 0.0, 0.0, 0.0};

    math::qnumber v_A = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber v_rho = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber v_g1 = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber v_g2 = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber v_x = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber v_y = {0.0, 0.0, 0.0, 0.0, 0.0};

    math::qnumber v_AA = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber v_rhorho = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber v_g1g1 = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber v_g2g2 = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber v_xx = {0.0, 0.0, 0.0, 0.0, 0.0};
    math::qnumber v_yy = {0.0, 0.0, 0.0, 0.0, 0.0};

    modelNumber(
        math::qnumber v,
        math::qnumber v_A,
        math::qnumber v_rho,
        math::qnumber v_g1,
        math::qnumber v_g2,
        math::qnumber v_x,
        math::qnumber v_y,
        math::qnumber v_AA,
        math::qnumber v_rhorho,
        math::qnumber v_g1g1,
        math::qnumber v_g2g2,
        math::qnumber v_xx,
        math::qnumber v_yy
    ) : v(v), v_A(v_A), v_rho(v_rho), v_g1(v_g1), v_g2(v_g2),
        v_x(v_x), v_y(v_y),
        v_AA(v_AA), v_rhorho(v_rhorho), v_g1g1(v_g1g1), v_g2g2(v_g2g2),
        v_xx(v_xx), v_yy(v_yy) {}

    // Define addition for modelNumber + modelNumber
    modelNumber operator+(const modelNumber& other) const {
        return modelNumber(
            this->v + other.v,
            this->v_A + other.v_A,
            this->v_rho + other.v_rho,
            this->v_g1 + other.v_g1,
            this->v_g2 + other.v_g2,
            this->v_x + other.v_x,
            this->v_y + other.v_y,
            this->v_AA + other.v_AA,
            this->v_rhorho + other.v_rhorho,
            this->v_g1g1 + other.v_g1g1,
            this->v_g2g2 + other.v_g2g2,
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
            this->v_g1 - other.v_g1,
            this->v_g2 - other.v_g2,
            this->v_x - other.v_x,
            this->v_y - other.v_y,
            this->v_AA - other.v_AA,
            this->v_rhorho - other.v_rhorho,
            this->v_g1g1 - other.v_g1g1,
            this->v_g2g2 - other.v_g2g2,
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
            -this->v_g1,
            -this->v_g2,
            -this->v_x,
            -this->v_y,
            -this->v_AA,
            -this->v_rhorho,
            -this->v_g1g1,
            -this->v_g2g2,
            -this->v_xx,
            -this->v_yy
        );
    }
};


class NgmixModel {
private:
    // Preventing copy (implement these if you need copy semantics)
    NgmixModel(const NgmixModel&) = delete;
    NgmixModel& operator=(const NgmixModel&) = delete;

    virtual frDeriv get_fr(
        math::qnumber r2
    ) const {
        return frDeriv(r2, r2, r2);
    };

public:
    math::qnumber A = {0.0, 0.0, 0.0, 0.0, 0.0};         // Amplitude
    math::qnumber rho = {0.0, 0.0, 0.0, 0.0, 0.0};       // convergence
    math::qnumber Gamma1 = {0.0, 0.0, 0.0, 0.0, 0.0};    // g1
    math::qnumber Gamma2 = {0.0, 0.0, 0.0, 0.0, 0.0};    // g2
    math::qnumber x0 = {0.0, 0.0, 0.0, 0.0, 0.0};        // position x
    math::qnumber y0 = {0.0, 0.0, 0.0, 0.0, 0.0};        // position y

    NgmixModel() {};

    void set_params(
        std::vector<math::qnumber> params
    ) {
        this->A = params[0];
        this->rho = params[1];
        this->Gamma1 = params[2];
        this->Gamma2 = params[3];
        this->x0 = params[4];
        this->y0 = params[5];
    };

    modelNumber get_r2(
        double x,
        double y
    ) const {

        // Center Shifting
        math::qnumber x_s = x - x0;
        math::qnumber y_s = y - y0;
        // Shearing
        math::qnumber x_transformed = (
            x_s * (1.0 - Gamma1) + y_s * Gamma2 * -1.0
        ) / rho;
        math::qnumber y_transformed = (
            x_s * Gamma2 * -1.0 + y_s * (1.0 + Gamma1)
        ) / rho;
        math::qnumber q = {0.0, 0.0, 0.0, 0.0, 0.0};
        modelNumber res{q, q, q, q, q, q, q, q, q, q, q, q, q};


        res.v = x_transformed * x_transformed + y_transformed * y_transformed;

        // First-order derivatives
        res.v_A = q;
        math::qnumber r1 = 2.0 / rho;
        math::qnumber r2 = r1 / rho;
        res.v_rho = -1.0 * r1 * res.v;
        res.v_g1 = r1 * (-x_s * x_transformed + y_s * y_transformed);
        res.v_g2 = r1 * (-y_s * x_transformed - x_s * y_transformed);
        res.v_x = r1 * (
            -(1.0 - Gamma1) * x_transformed + Gamma2 * y_transformed
        );
        res.v_y = r1 * (
            (Gamma2) * x_transformed - (1.0 + Gamma1) * y_transformed
        );
        math::qnumber r2_s = x_s * x_s + y_s * y_s;

        // Second-order derivatives
        res.v_AA = q;
        res.v_rhorho = 3.0 * r2 * res.v;
        res.v_g1g1 = r2 * r2_s;
        res.v_g2g2 = r2 * r2_s;
        res.v_xx = r2 * (
            (1.0 - Gamma1) * (1.0 - Gamma1) + Gamma2 * Gamma2
        );
        res.v_yy = r2 * (
            (1.0 + Gamma1) * (1.0 + Gamma1) + Gamma2 * Gamma2
        );
        return res;
    };

    modelNumber model(
        double x, double y
    ) const {

        modelNumber r2 = this->get_r2(x, y);
        frDeriv fr =  this->get_fr(r2.v);
        math::qnumber q = {0.0, 0.0, 0.0, 0.0, 0.0};
        modelNumber res{q, q, q, q, q, q, q, q, q, q, q, q, q};
        res.v = fr.fr;

        // First-order derivatives
        res.v_A = fr.fr / this->A;
        res.v_rho = fr.dfr * r2.v_rho;
        res.v_g1 = fr.dfr * r2.v_g1;
        res.v_g2 = fr.dfr * r2.v_g2;
        res.v_x = fr.dfr * r2.v_x;
        res.v_y = fr.dfr * r2.v_y;

        // Second-order derivatives
        res.v_AA = q;
        res.v_rhorho = fr.ddfr * r2.v_rho * r2.v_rho + fr.dfr * r2.v_rhorho;
        res.v_g1g1 = fr.ddfr * r2.v_g1 * r2.v_g1 + fr.dfr * r2.v_g1g1;
        res.v_g2g2 = fr.ddfr * r2.v_g2 * r2.v_g2 + fr.dfr * r2.v_g2g2;
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

        math::qnumber q = {0.0, 0.0, 0.0, 0.0, 0.0};
        modelNumber res{q, q, q, q, q, q, q, q, q, q, q, q, q};

        res.v = 0.5 * residual * residual / variance_val;

        // First-order derivatives
        math::qnumber tmp = -1.0 * residual / variance_val;
        res.v_A =  tmp * theory_val.v_A ;
        res.v_rho = tmp * theory_val.v_rho;
        res.v_g1 = tmp * theory_val.v_g1;
        res.v_g2 = tmp * theory_val.v_g2;
        res.v_x = tmp * theory_val.v_x;
        res.v_y = tmp * theory_val.v_y;

        // Second-order derivatives
        res.v_AA = (
            (theory_val.v_A * theory_val.v_A) / variance_val
            + tmp * theory_val.v_AA
        );
        res.v_rhorho = (
            (theory_val.v_rho * theory_val.v_rho) / variance_val
            + tmp * theory_val.v_rhorho
        );
        res.v_g1g1 = (
            (theory_val.v_g1 * theory_val.v_g1) / variance_val
            + tmp * theory_val.v_g1g1
        );
        res.v_g2g2 = (
            (theory_val.v_g2 * theory_val.v_g2) / variance_val
            + tmp * theory_val.v_g2g2
        );
        res.v_xx = (
            (theory_val.v_x * theory_val.v_x) / variance_val
            + tmp * theory_val.v_xx
        );
        res.v_yy = (
            (theory_val.v_y * theory_val.v_y) / variance_val
            + tmp * theory_val.v_yy
        );
        return res;
    };

    NgmixModel(NgmixModel&& other) noexcept = default;
    NgmixModel& operator=(NgmixModel&& other) noexcept = default;

    virtual ~NgmixModel() = default;

};


/// NgmixGaussian Function
class NgmixGaussian : public NgmixModel {
private:
    double sigma;
    double _p0;
    frDeriv get_fr(
        math::qnumber r2
    ) const {
        math::qnumber fr = this->A * math::exp(r2 * this->_p0);
        math::qnumber dfr = fr * this->_p0;
        math::qnumber ddfr = dfr * this->_p0;
        return frDeriv(fr, dfr, ddfr);
    };
public:
    // NgmixGaussian Profile
    NgmixGaussian(double sigma) : sigma(sigma) {
        _p0 = -1.0 / (2 * sigma * sigma);
    };

};

}
}
#endif // ANACAL_NGMIX_RMODEL_H
