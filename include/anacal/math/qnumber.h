#ifndef ANACAL_MATH_QN_H
#define ANACAL_MATH_QN_H

#include <cmath>
#include <ostream>

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
        double v, double g1, double g2, double x1, double x2
    )
        : v(v), g1(g1), g2(g2), x1(x1), x2(x2) {}

    // Define addition for qnumber + qnumber
    qnumber operator+(const qnumber& other) const {
        return qnumber(
            this->v + other.v,
            this->g1 + other.g1,
            this->g2 + other.g2,
            this->x1 + other.x1,
            this->x2 + other.x2
        );
    }

    // Define subtraction for qnumber - qnumber
    qnumber operator-(const qnumber& other) const {
        return qnumber(
            this->v - other.v,
            this->g1 - other.g1,
            this->g2 - other.g2,
            this->x1 - other.x1,
            this->x2 - other.x2
        );
    }

    // Define unary negation for -qnumber
    qnumber operator-() const {
        return qnumber(
            -this->v,
            -this->g1,
            -this->g2,
            -this->x1,
            -this->x2
        );
    }

    // Define multiplication for qnumber * qnumber
    qnumber operator*(const qnumber& other) const {
        return qnumber(
            this->v * other.v,
            this->g1 * other.v + this->v * other.g1,
            this->g2 * other.v + this->v * other.g2,
            this->x1 * other.v + this->v * other.x1,
            this->x2 * other.v + this->v * other.x2
        );
    }

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
    }

    // Friend function for addition: qnumber + T
    template <typename T>
    friend qnumber operator+(const qnumber& lhs, T rhs) {
        return qnumber(
            lhs.v + rhs,
            lhs.g1,
            lhs.g2,
            lhs.x1,
            lhs.x2
        );
    }

    // Friend functions for addition: T + qnumber
    template <typename T>
    friend qnumber operator+(T lhs, const qnumber& rhs) {
        return qnumber(
            lhs + rhs.v,
            rhs.g1,
            rhs.g2,
            rhs.x1,
            rhs.x2
        );
    }

    // Friend function for subtraction: qnumber - T
    template <typename T>
    friend qnumber operator-(const qnumber& lhs, T rhs) {
        return qnumber(
            lhs.v - rhs,
            lhs.g1,
            lhs.g2,
            lhs.x1,
            lhs.x2
        );
    }

    // Friend functions for subtraction: T - qnumber
    template <typename T>
    friend qnumber operator-(T lhs, const qnumber& rhs) {
        return qnumber(
            lhs - rhs.v,
            -rhs.g1,
            -rhs.g2,
            -rhs.x1,
            -rhs.x2
        );
    }

    // Friend function for multiplication: qnumber * T
    template <typename T>
    friend qnumber operator*(const qnumber& lhs, T rhs) {
        return qnumber(
            lhs.v * rhs,
            lhs.g1 * rhs,
            lhs.g2 * rhs,
            lhs.x1 * rhs,
            lhs.x2 * rhs
        );
    }

    // Friend functions for multiplication: T * qnumber
    template <typename T>
    friend qnumber operator*(T lhs, const qnumber& rhs) {
        return qnumber(
            lhs * rhs.v,
            lhs * rhs.g1,
            lhs * rhs.g2,
            lhs * rhs.x1,
            lhs * rhs.x2
        );
    }

    // Friend function for division: qnumber / T
    template <typename T>
    friend qnumber operator/(const qnumber& lhs, T rhs) {
        return qnumber(
            lhs.v / rhs,
            lhs.g1 / rhs,
            lhs.g2 / rhs,
            lhs.x1 / rhs,
            lhs.x2 / rhs
        );
    }

    // Friend functions for division: T / qnumber
    template <typename T>
    friend qnumber operator/(T lhs, const qnumber& rhs) {
        double f = -1.0 / (rhs.v * rhs.v);
        return qnumber(
            lhs / rhs.v,
            lhs * rhs.g1 * f,
            lhs * rhs.g2 * f,
            lhs * rhs.x1 * f,
            lhs * rhs.x2 * f
        );
    }

    // Stream insertion operator for printing
    friend std::ostream& operator<<(std::ostream& os, const qnumber& q) {
        os << "v: " << q.v << ", g1: " << q.g1 << ", g2: " << q.g2
           << ", x1: " << q.x1 << ", x2: " << q.x2;
        return os;
    }
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

inline py::array_t<double> qnumber_to_array(
    const qnumber& qn
) {
    py::array_t<double> out(5);
    auto out_r = out.mutable_unchecked<1>();
    out_r(0) = qn.v;
    out_r(1) = qn.g1;
    out_r(2) = qn.g2;
    out_r(3) = qn.x1;
    out_r(4) = qn.x2;
    return out;
};  // qnumber_to_array


inline qnumber array_to_qnumber(
    const py::array_t<double>& array
) {
    auto array_r = array.unchecked<1>();
    return qnumber(
        array_r(0),
        array_r(1),
        array_r(2),
        array_r(3),
        array_r(4)
    );
}; // array_to_qnumber




} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_QN_H
