#ifndef ANACAL_MATH_QN_H
#define ANACAL_MATH_QN_H

#include <cmath>
#include <ostream>

namespace anacal {
namespace math {


struct qnumber {
    double v, g1, g2, x1, x2;

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
            (this->g1 * other.v - this->v * other.g1) * f,
            (this->g2 * other.v - this->v * other.g2) * f,
            (this->x1 * other.v - this->v * other.x1) * f,
            (this->x2 * other.v - this->v * other.x2) * f
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

    // Templated comparison operators
    template <typename T>
    friend bool operator==(const qnumber& lhs, T rhs) {
        return lhs.v == rhs;
    }

    template <typename T>
    friend bool operator==(T lhs, const qnumber& rhs) {
        return lhs == rhs.v;
    }

    template <typename T>
    friend bool operator!=(const qnumber& lhs, T rhs) {
        return lhs.v != rhs;
    }

    template <typename T>
    friend bool operator!=(T lhs, const qnumber& rhs) {
        return lhs != rhs.v;
    }

    template <typename T>
    friend bool operator<(const qnumber& lhs, T rhs) {
        return lhs.v < rhs;
    }

    template <typename T>
    friend bool operator<(T lhs, const qnumber& rhs) {
        return lhs < rhs.v;
    }

    template <typename T>
    friend bool operator<=(const qnumber& lhs, T rhs) {
        return lhs.v <= rhs;
    }

    template <typename T>
    friend bool operator<=(T lhs, const qnumber& rhs) {
        return lhs <= rhs.v;
    }

    template <typename T>
    friend bool operator>(const qnumber& lhs, T rhs) {
        return lhs.v > rhs;
    }

    template <typename T>
    friend bool operator>(T lhs, const qnumber& rhs) {
        return lhs > rhs.v;
    }

    template <typename T>
    friend bool operator>=(const qnumber& lhs, T rhs) {
        return lhs.v >= rhs;
    }

    template <typename T>
    friend bool operator>=(T lhs, const qnumber& rhs) {
        return lhs >= rhs.v;
    }

    // Stream insertion operator for printing
    friend std::ostream& operator<<(std::ostream& os, const qnumber& q) {
        os << "v: " << q.v << ", g1: " << q.g1 << ", g2: " << q.g2
           << ", x1: " << q.x1 << ", x2: " << q.x2;
        return os;
    }
};


} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_QN_H
