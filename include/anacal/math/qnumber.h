#ifndef ANACAL_MATH_QN_H
#define ANACAL_MATH_QN_H

#include "../stdafx.h"

// Class for Quintuple number

namespace anacal {
namespace math {

struct qnumber {
    double v  = 0.0;
    double g1 = 0.0;
    double g2 = 0.0;
    double x1 = 0.0;
    double x2 = 0.0;

    qnumber() = default;

    qnumber(
        double v
    )
        : v(v) {}

    qnumber(
        double v, double g1, double g2
    )
        : v(v), g1(g1), g2(g2) {}

    qnumber(
        double v, double g1, double g2, double x1, double x2
    )
        : v(v), g1(g1), g2(g2), x1(x1), x2(x2) {}

    qnumber(const std::array<double, 5>& data) {
        this->v = data[0];
        this->g1 = data[1];
        this->g2 = data[2];
        this->x1 = data[3];
        this->x2 = data[4];
    };

    // Define addition for qnumber + qnumber
    qnumber operator+(const qnumber& other) const {
        return qnumber(
            this->v + other.v,
            this->g1 + other.g1,
            this->g2 + other.g2,
            this->x1 + other.x1,
            this->x2 + other.x2
        );
    };

    // Define subtraction for qnumber - qnumber
    qnumber operator-(const qnumber& other) const {
        return qnumber(
            this->v - other.v,
            this->g1 - other.g1,
            this->g2 - other.g2,
            this->x1 - other.x1,
            this->x2 - other.x2
        );
    };

    // Define unary negation for -qnumber
    qnumber operator-() const {
        return qnumber(
            -this->v,
            -this->g1,
            -this->g2,
            -this->x1,
            -this->x2
        );
    };

    // Define multiplication for qnumber * qnumber
    qnumber operator*(const qnumber& other) const {
        return qnumber(
            this->v * other.v,
            this->g1 * other.v + this->v * other.g1,
            this->g2 * other.v + this->v * other.g2,
            this->x1 * other.v + this->v * other.x1,
            this->x2 * other.v + this->v * other.x2
        );
    };

    // Define division for qnumber / qnumber
    qnumber operator/(const qnumber& other) const {
        double f = 1.0 / (other.v * other.v);
        return qnumber(
            this->v / other.v,
            (other.v * this->g1 - this->v * other.g1) * f,
            (other.v * this->g2 - this->v * other.g2) * f,
            (other.v * this->x1 - this->v * other.x1) * f,
            (other.v * this->x2 - this->v * other.x2) * f
        );
    };

    // Friend function for addition: qnumber + double
    friend qnumber operator+(const qnumber& lhs, double rhs) {
        return qnumber(
            lhs.v + rhs,
            lhs.g1,
            lhs.g2,
            lhs.x1,
            lhs.x2
        );
    };

    // Friend functions for addition: double + qnumber
    friend qnumber operator+(double lhs, const qnumber& rhs) {
        return qnumber(
            lhs + rhs.v,
            rhs.g1,
            rhs.g2,
            rhs.x1,
            rhs.x2
        );
    };

    // Friend function for subtraction: qnumber - double
    friend qnumber operator-(const qnumber& lhs, double rhs) {
        return qnumber(
            lhs.v - rhs,
            lhs.g1,
            lhs.g2,
            lhs.x1,
            lhs.x2
        );
    };

    // Friend functions for subtraction: double - qnumber
    friend qnumber operator-(double lhs, const qnumber& rhs) {
        return qnumber(
            lhs - rhs.v,
            -rhs.g1,
            -rhs.g2,
            -rhs.x1,
            -rhs.x2
        );
    };

    // Friend function for multiplication: qnumber * double
    friend qnumber operator*(const qnumber& lhs, double rhs) {
        return qnumber(
            lhs.v * rhs,
            lhs.g1 * rhs,
            lhs.g2 * rhs,
            lhs.x1 * rhs,
            lhs.x2 * rhs
        );
    };

    // Friend functions for multiplication: double * qnumber
    friend qnumber operator*(double lhs, const qnumber& rhs) {
        return qnumber(
            lhs * rhs.v,
            lhs * rhs.g1,
            lhs * rhs.g2,
            lhs * rhs.x1,
            lhs * rhs.x2
        );
    };

    // Friend function for division: qnumber / double
    friend qnumber operator/(const qnumber& lhs, double rhs) {
        return qnumber(
            lhs.v / rhs,
            lhs.g1 / rhs,
            lhs.g2 / rhs,
            lhs.x1 / rhs,
            lhs.x2 / rhs
        );
    };

    // Friend functions for division: double / qnumber
    friend qnumber operator/(double lhs, const qnumber& rhs) {
        double f = -1.0 / (rhs.v * rhs.v);
        return qnumber(
            lhs / rhs.v,
            lhs * rhs.g1 * f,
            lhs * rhs.g2 * f,
            lhs * rhs.x1 * f,
            lhs * rhs.x2 * f
        );
    };

    // to array
    py::array_t<double> to_array() const {
        auto result = py::array_t<double>(5);
        auto res_r = result.mutable_unchecked<1>();
        res_r(0) = this->v;
        res_r(1) = this->g1;
        res_r(2) = this->g2;
        res_r(3) = this->x1;
        res_r(4) = this->x2;
        return result;
    };

    qnumber decentralize(double dx1, double dx2) const {
        // (dx1, dx2) is the position of the source wrt center of block
        qnumber result = *this;
        result.g1 = this->g1 - dx1 * this->x1 + dx2 * this->x2;
        result.g2 = this->g2 - dx1 * this->x2 - dx2 * this->x1;
        return result;
    };

    qnumber centralize(double dx1, double dx2) const {
        // (dx1, dx2) is the position of the source wrt center of block
        qnumber result = *this;
        result.g1 = this->g1 + dx1 * this->x1 - dx2 * this->x2;
        result.g2 = this->g2 + dx1 * this->x2 + dx2 * this->x1;
        return result;
    };

    // Stream insertion operator for printing
    friend std::ostream& operator<<(std::ostream& os, const qnumber& q) {
        os << "v: " << q.v << ", g1: " << q.g1 << ", g2: " << q.g2
           << ", x1: " << q.x1 << ", x2: " << q.x2;
        return os;
    };
};


inline qnumber exp(
    const qnumber& qn
) {
    double expv = std::exp(qn.v);
    return qnumber(
        expv,
        expv * qn.g1,
        expv * qn.g2,
        expv * qn.x1,
        expv * qn.x2
    );
}; // exponential function

inline qnumber tanh(
    const qnumber& qn
) {
    double tan = std::tanh(qn.v);
    double dtan = 1.0 - pow(tan, 2.0);
    return qnumber(
        tan,
        dtan * qn.g1,
        dtan * qn.g2,
        dtan * qn.x1,
        dtan * qn.x2
    );
}; // tanh function

inline qnumber sin(
    const qnumber& qn
) {
    double sin = std::sin(qn.v);
    double dsin = std::cos(qn.v);
    return qnumber(
        sin,
        dsin * qn.g1,
        dsin * qn.g2,
        dsin * qn.x1,
        dsin * qn.x2
    );
}; // sine function

inline qnumber cos(
    const qnumber& qn
) {
    double cos = std::cos(qn.v);
    double dcos = -std::sin(qn.v);
    return qnumber(
        cos,
        dcos * qn.g1,
        dcos * qn.g2,
        dcos * qn.x1,
        dcos * qn.x2
    );
}; // cosine function

inline qnumber atan2(
    const qnumber& qn1,
    const qnumber& qn2
) {
    double v0 = std::atan2(qn1.v, qn2.v);
    double denom = qn1.v * qn1.v + qn2.v * qn2.v;
    double d1 = qn2.v / denom;
    double d2 = -qn1.v / denom;
    return qnumber(
        v0,
        d1 * qn1.g1 + d2 * qn2.g1,
        d1 * qn1.g2 + d2 * qn2.g2,
        d1 * qn1.x1 + d2 * qn2.x1,
        d1 * qn1.x2 + d2 * qn2.x2
    );
}; // arctan2 function

inline qnumber pow(
    const qnumber& qn,
    double n
) {
    double tmp0 = std::pow(qn.v, n - 1);
    double tmp = n * tmp0;
    return qnumber(
        tmp0 * qn.v,
        tmp * qn.g1,
        tmp * qn.g2,
        tmp * qn.x1,
        tmp * qn.x2
    );
}; // power function

inline qnumber sqrt(
    const qnumber& qn
) {
    double tmp0 = std::sqrt(qn.v);
    double tmp = 0.5 / tmp0;
    return qnumber(
        tmp0,
        tmp * qn.g1,
        tmp * qn.g2,
        tmp * qn.x1,
        tmp * qn.x2
    );
}; // sqrt function

inline constexpr int N_LOOK  = 33;
/**
* Compile-time exponent lookup table for i in [MIN_VAL, MAX_VAL].
*/
inline constexpr double EXP_LOOKUP[N_LOOK] = { 3.059023205018258e-07,
    5.04347662567888e-07, 8.315287191035679e-07, 1.3709590863840845e-06,
    2.2603294069810542e-06, 3.726653172078671e-06, 6.14421235332821e-06,
    1.013009359863071e-05, 1.670170079024566e-05, 2.7536449349747158e-05,
    4.5399929762484854e-05, 7.48518298877006e-05, 0.00012340980408667956,
    0.00020346836901064417, 0.00033546262790251185, 0.0005530843701478336,
    0.0009118819655545162, 0.0015034391929775724, 0.0024787521766663585,
    0.004086771438464067, 0.006737946999085467, 0.011108996538242306,
    0.01831563888873418, 0.0301973834223185, 0.049787068367863944,
    0.0820849986238988, 0.1353352832366127, 0.22313016014842982,
    0.36787944117144233, 0.6065306597126334, 1.0, 1.6487212707001282,
    2.718281828459045
};

inline qnumber exp6(
    const qnumber& qn
) {
    int ival = static_cast<int>((qn.v + 15) * 2);
    double f = qn.v - (-15.0 + 0.5 * ival);
    // get the base exp from the precomputed table
    double expv = EXP_LOOKUP[ival];
    // multiply by the polynomial factor
    expv *= (
        720.0 + f * (
            720.0 + f * (
                360.0 + f * (
                    120.0 + f * (
                        30.0 + f * (
                            6.0 + f
                        )
                    )
                )
            )
        )
    ) * 0.001388888888888889;
    return qnumber(
        expv,
        expv * qn.g1,
        expv * qn.g2,
        expv * qn.x1,
        expv * qn.x2
    );
};


} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_QN_H
