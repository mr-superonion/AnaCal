#ifndef ANACAL_MATH_LOSS_H
#define ANACAL_MATH_LOSS_H

#include "../stdafx.h"
#include "qnumber.h"

namespace anacal {
namespace math {

// Every qnumber field of lossNumber, once: the operators below are
// generated from this list so that a field added here cannot be
// forgotten in one of them.  v is the loss; v_X its gradient in model
// parameter X (F: flux; mxx, myy, mxy: intrinsic covariance
// components; x1, x2: centre); v_XX and v_XY the Gauss-Newton
// curvature (diagonal and cross terms).
#define ANACAL_LOSS_FIELDS(X) \
    X(v) X(v_in) \
    X(v_F) X(v_mxx) X(v_myy) X(v_mxy) X(v_x1) X(v_x2) \
    X(v_FF) X(v_mxxmxx) X(v_myymyy) X(v_mxymxy) X(v_x1x1) X(v_x2x2) \
    X(v_mxxmyy) X(v_mxxmxy) X(v_myymxy) \
    X(v_mxxx1) X(v_mxxx2) X(v_myyx1) X(v_myyx2) X(v_mxyx1) X(v_mxyx2) \
    X(v_x1x2) \
    X(v_Fmxx) X(v_Fmyy) X(v_Fmxy) X(v_Fx1) X(v_Fx2)

struct lossNumber {
    // value with derivatives to Gaussian model parameters
#define ANACAL_LOSS_DECLARE(name) math::qnumber name;
    ANACAL_LOSS_FIELDS(ANACAL_LOSS_DECLARE)
#undef ANACAL_LOSS_DECLARE
    // v is the chi2 of the whole fitting window; v_in and n_pix are the
    // chi2 and the pixel count of the pixels the model actually covers
    // (inside its apodisation edge) -- what the misfit damping in
    // update_model_params looks at, so that a neighbour elsewhere in
    // the window does not count as misfit of this source
    double n_pix = 0.0;

    lossNumber() = default;

    lossNumber operator+(const lossNumber& other) const {
        lossNumber out;
#define ANACAL_LOSS_ADD(name) out.name = this->name + other.name;
        ANACAL_LOSS_FIELDS(ANACAL_LOSS_ADD)
#undef ANACAL_LOSS_ADD
        out.n_pix = this->n_pix + other.n_pix;
        return out;
    }

    lossNumber operator-(const lossNumber& other) const {
        lossNumber out;
#define ANACAL_LOSS_SUB(name) out.name = this->name - other.name;
        ANACAL_LOSS_FIELDS(ANACAL_LOSS_SUB)
#undef ANACAL_LOSS_SUB
        out.n_pix = this->n_pix - other.n_pix;
        return out;
    }

    lossNumber operator-() const {
        lossNumber out;
#define ANACAL_LOSS_NEG(name) out.name = -this->name;
        ANACAL_LOSS_FIELDS(ANACAL_LOSS_NEG)
#undef ANACAL_LOSS_NEG
        out.n_pix = -this->n_pix;
        return out;
    }

    inline void reset() {
#define ANACAL_LOSS_RESET(name) this->name = math::qnumber(0.0);
        ANACAL_LOSS_FIELDS(ANACAL_LOSS_RESET)
#undef ANACAL_LOSS_RESET
        this->n_pix = 0.0;
    };
};


} // end of math
} // end of anacal

#endif // ANACAL_MATH_LOSS_H
