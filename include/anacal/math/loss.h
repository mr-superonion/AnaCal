#ifndef ANACAL_MATH_LOSS_H
#define ANACAL_MATH_LOSS_H

#include "../stdafx.h"
#include "qnumber.h"

namespace anacal {
namespace math {

struct lossNumber {
    // value with derivatives to Gaussian model parameters
    math::qnumber v, v_F, v_t, v_e1, v_e2, v_x1, v_x2;
    math::qnumber v_FF, v_tt, v_e1e1, v_e2e2, v_x1x1, v_x2x2;

    lossNumber() = default;

    lossNumber(
        math::qnumber v,
        math::qnumber v_F,
        math::qnumber v_t,
        math::qnumber v_e1,
        math::qnumber v_e2,
        math::qnumber v_x1,
        math::qnumber v_x2,
        math::qnumber v_FF,
        math::qnumber v_tt,
        math::qnumber v_e1e1,
        math::qnumber v_e2e2,
        math::qnumber v_x1x1,
        math::qnumber v_x2x2
    ) : v(v), v_F(v_F), v_t(v_t), v_e1(v_e1), v_e2(v_e2),
        v_x1(v_x1), v_x2(v_x2),
        v_FF(v_FF), v_tt(v_tt), v_e1e1(v_e1e1), v_e2e2(v_e2e2),
        v_x1x1(v_x1x1), v_x2x2(v_x2x2) {}

    // Define addition for lossNumber + lossNumber
    lossNumber operator+(const lossNumber& other) const {
        return lossNumber(
            this->v + other.v,
            this->v_F + other.v_F,
            this->v_t + other.v_t,
            this->v_e1 + other.v_e1,
            this->v_e2 + other.v_e2,
            this->v_x1 + other.v_x1,
            this->v_x2 + other.v_x2,
            this->v_FF + other.v_FF,
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
            this->v_F - other.v_F,
            this->v_t - other.v_t,
            this->v_e1 - other.v_e1,
            this->v_e2 - other.v_e2,
            this->v_x1 - other.v_x1,
            this->v_x2 - other.v_x2,
            this->v_FF - other.v_FF,
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
            -this->v_F,
            -this->v_t,
            -this->v_e1,
            -this->v_e2,
            -this->v_x1,
            -this->v_x2,
            -this->v_FF,
            -this->v_tt,
            -this->v_e1e1,
            -this->v_e2e2,
            -this->v_x1x1,
            -this->v_x2x2
        );
    }
};


} // end of math
} // end of anacal

#endif // ANACAL_MATH_LOSS_H
