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
        double v
    )
        : v(v) {}

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
    const qnumber& tn
) {
    double expv = std::exp(tn.v);
    return qnumber(
        expv,
        expv * tn.g1,
        expv * tn.g2,
        expv * tn.x1,
        expv * tn.x2
    );
}; // exponential function

inline qnumber tanh(
    const qnumber& tn
) {
    double tan = std::tanh(tn.v);
    double dtan = 1.0 - pow(tan, 2.0);
    return qnumber(
        tan,
        dtan * tn.g1,
        dtan * tn.g2,
        dtan * tn.x1,
        dtan * tn.x2
    );
}; // tanh function

inline qnumber pow(
    const qnumber& tn,
    double n
) {
    double tmp0 = std::pow(tn.v, n - 1);
    double tmp = n * tmp0;
    return qnumber(
        tmp0 * tn.v,
        tmp * tn.g1,
        tmp * tn.g2,
        tmp * tn.x1,
        tmp * tn.x2
    );
}; // power function



} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_QN_H
