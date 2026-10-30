#ifndef ANACAL_MATH_QN_H
#define ANACAL_MATH_QN_H

#include "../stdafx.h"

// Quintuple / triple number: a value with its tangents.  qnumber_t<T, NT>
// carries NT = 4 tangents (g1, g2: shear; x1, x2: position -- needed when a
// cell is measured as a whole) or NT = 2 (g1, g2 only: enough for a stamp
// measured about a fixed centre), in float or double.  ``qnumber`` is the
// Task's type.  Every operator keeps the floating-point operation order of
// the original double quintuple, so qnumber_t<double, 4> is bit-identical
// to it.

namespace anacal {
namespace math {

// The position tangents, present only with four tangents.
template <typename T, int NT>
struct qposition {
    T x1 = 0;
    T x2 = 0;
};

template <typename T>
struct qposition<T, 2> {};

template <typename T, int NT = 4>
struct qnumber_t : qposition<T, NT> {
    static_assert(std::is_floating_point<T>::value, "qnumber_t: T must be float or double");
    static_assert(NT == 2 || NT == 4, "qnumber_t: 2 or 4 tangents");
    using value_type = T;
    static constexpr int ntangent = NT;
    static constexpr int ncomp = NT + 1;
    static constexpr bool has_position = (NT == 4);

    T v  = 0;
    T g1 = 0;
    T g2 = 0;

    qnumber_t() = default;

    qnumber_t(
        T v
    )
        : v(v) {}

    qnumber_t(
        T v, T g1, T g2
    )
        : v(v), g1(g1), g2(g2) {}

    template <int N = NT, std::enable_if_t<N == 4, int> = 0>
    qnumber_t(
        T v, T g1, T g2, T x1, T x2
    )
        : v(v), g1(g1), g2(g2) {
        this->x1 = x1;
        this->x2 = x2;
    }

    qnumber_t(const std::array<T, ncomp>& data) {
        this->v = data[0];
        this->g1 = data[1];
        this->g2 = data[2];
        if constexpr (NT == 4) {
            this->x1 = data[3];
            this->x2 = data[4];
        }
    };

    // component k in the order (v, g1, g2, x1, x2)
    T comp(int k) const {
        switch (k) {
        case 0: return this->v;
        case 1: return this->g1;
        case 2: return this->g2;
        default:
            if constexpr (NT == 4) {
                return (k == 3) ? this->x1 : this->x2;
            } else {
                return T(0);
            }
        }
    }

    T& comp(int k) {
        switch (k) {
        case 0: return this->v;
        case 1: return this->g1;
        case 2: return this->g2;
        default:
            if constexpr (NT == 4) {
                return (k == 3) ? this->x1 : this->x2;
            } else {
                return this->g2;  // unreachable for a valid k
            }
        }
    }

    // Define addition for qnumber + qnumber
    qnumber_t operator+(const qnumber_t& other) const {
        qnumber_t r(this->v + other.v, this->g1 + other.g1, this->g2 + other.g2);
        if constexpr (NT == 4) {
            r.x1 = this->x1 + other.x1;
            r.x2 = this->x2 + other.x2;
        }
        return r;
    };

    // Define subtraction for qnumber - qnumber
    qnumber_t operator-(const qnumber_t& other) const {
        qnumber_t r(this->v - other.v, this->g1 - other.g1, this->g2 - other.g2);
        if constexpr (NT == 4) {
            r.x1 = this->x1 - other.x1;
            r.x2 = this->x2 - other.x2;
        }
        return r;
    };

    // Define unary negation for -qnumber
    qnumber_t operator-() const {
        qnumber_t r(-this->v, -this->g1, -this->g2);
        if constexpr (NT == 4) {
            r.x1 = -this->x1;
            r.x2 = -this->x2;
        }
        return r;
    };

    // Define multiplication for qnumber * qnumber
    qnumber_t operator*(const qnumber_t& other) const {
        qnumber_t r(
            this->v * other.v,
            this->g1 * other.v + this->v * other.g1,
            this->g2 * other.v + this->v * other.g2
        );
        if constexpr (NT == 4) {
            r.x1 = this->x1 * other.v + this->v * other.x1;
            r.x2 = this->x2 * other.v + this->v * other.x2;
        }
        return r;
    };

    // Define division for qnumber / qnumber
    qnumber_t operator/(const qnumber_t& other) const {
        T f = T(1.0) / (other.v * other.v);
        qnumber_t r(
            this->v / other.v,
            (other.v * this->g1 - this->v * other.g1) * f,
            (other.v * this->g2 - this->v * other.g2) * f
        );
        if constexpr (NT == 4) {
            r.x1 = (other.v * this->x1 - this->v * other.x1) * f;
            r.x2 = (other.v * this->x2 - this->v * other.x2) * f;
        }
        return r;
    };

    // Friend function for addition: qnumber + double
    friend qnumber_t operator+(const qnumber_t& lhs, double rhs) {
        qnumber_t r = lhs;
        r.v = lhs.v + T(rhs);
        return r;
    };

    // Friend functions for addition: double + qnumber
    friend qnumber_t operator+(double lhs, const qnumber_t& rhs) {
        qnumber_t r = rhs;
        r.v = T(lhs) + rhs.v;
        return r;
    };

    // Friend function for subtraction: qnumber - double
    friend qnumber_t operator-(const qnumber_t& lhs, double rhs) {
        qnumber_t r = lhs;
        r.v = lhs.v - T(rhs);
        return r;
    };

    // Friend functions for subtraction: double - qnumber
    friend qnumber_t operator-(double lhs, const qnumber_t& rhs) {
        qnumber_t r = -rhs;
        r.v = T(lhs) - rhs.v;
        return r;
    };

    // Friend function for multiplication: qnumber * double
    friend qnumber_t operator*(const qnumber_t& lhs, double rhs) {
        const T s = T(rhs);
        qnumber_t r(lhs.v * s, lhs.g1 * s, lhs.g2 * s);
        if constexpr (NT == 4) {
            r.x1 = lhs.x1 * s;
            r.x2 = lhs.x2 * s;
        }
        return r;
    };

    // Friend functions for multiplication: double * qnumber
    friend qnumber_t operator*(double lhs, const qnumber_t& rhs) {
        const T s = T(lhs);
        qnumber_t r(s * rhs.v, s * rhs.g1, s * rhs.g2);
        if constexpr (NT == 4) {
            r.x1 = s * rhs.x1;
            r.x2 = s * rhs.x2;
        }
        return r;
    };

    // Friend function for division: qnumber / double
    friend qnumber_t operator/(const qnumber_t& lhs, double rhs) {
        const T s = T(rhs);
        qnumber_t r(lhs.v / s, lhs.g1 / s, lhs.g2 / s);
        if constexpr (NT == 4) {
            r.x1 = lhs.x1 / s;
            r.x2 = lhs.x2 / s;
        }
        return r;
    };

    // Friend functions for division: double / qnumber
    friend qnumber_t operator/(double lhs, const qnumber_t& rhs) {
        const T s = T(lhs);
        T f = T(-1.0) / (rhs.v * rhs.v);
        qnumber_t r(s / rhs.v, s * rhs.g1 * f, s * rhs.g2 * f);
        if constexpr (NT == 4) {
            r.x1 = s * rhs.x1 * f;
            r.x2 = s * rhs.x2 * f;
        }
        return r;
    };

    // to array
    py::array_t<T> to_array() const {
        auto result = py::array_t<T>(ncomp);
        auto res_r = result.template mutable_unchecked<1>();
        for (int k = 0; k < ncomp; ++k) {
            res_r(k) = this->comp(k);
        }
        return result;
    };

    // In-place: only g1/g2 change, and they are computed from x1/x2 (not
    // from each other), so the write order is alias-free.
    // One signed reference shift: the shear responses g1/g2 pick up
    // position-coupling terms when the reference point of the
    // coordinates moves by (dx1, dx2).  Both wrappers below call this
    // with opposite signs; (dx1, dx2) is always the source position wrt
    // the cell center.
    template <int N = NT, std::enable_if_t<N == 4, int> = 0>
    void shift_reference(double dx1, double dx2) {
        this->g1 = this->g1 + T(dx1) * this->x1 - T(dx2) * this->x2;
        this->g2 = this->g2 + T(dx1) * this->x2 + T(dx2) * this->x1;
    };

    // Positive sign: shifts the reference point to the CELL CENTER
    // (applied before per-cell fitting).
    template <int N = NT, std::enable_if_t<N == 4, int> = 0>
    void centralize(double dx1, double dx2) {
        this->shift_reference(dx1, dx2);
    };

    // Opposite sign: shifts the reference point back to the DETECTION
    // PEAK (applied after per-cell fitting).
    template <int N = NT, std::enable_if_t<N == 4, int> = 0>
    void decentralize(double dx1, double dx2) {
        this->shift_reference(-dx1, -dx2);
    };

    // Stream insertion operator for printing
    friend std::ostream& operator<<(std::ostream& os, const qnumber_t& q) {
        os << "v: " << q.v << ", g1: " << q.g1 << ", g2: " << q.g2;
        if constexpr (NT == 4) {
            os << ", x1: " << q.x1 << ", x2: " << q.x2;
        }
        return os;
    };
};

// The Task's number: double, four tangents.
using qnumber = qnumber_t<double, 4>;

// f(x) with derivative df at x.v applied to all tangent slots (the chain
// rule of every elementary function below).
template <typename T, int NT>
inline qnumber_t<T, NT> chain(T f, T df, const qnumber_t<T, NT>& x) {
    qnumber_t<T, NT> r(f, df * x.g1, df * x.g2);
    if constexpr (NT == 4) {
        r.x1 = df * x.x1;
        r.x2 = df * x.x2;
    }
    return r;
}

template <typename T, int NT>
inline qnumber_t<T, NT> exp(
    const qnumber_t<T, NT>& qn
) {
    T expv = std::exp(qn.v);
    return chain(expv, expv, qn);
}; // exponential function

template <typename T, int NT>
inline qnumber_t<T, NT> log(
    const qnumber_t<T, NT>& qn
) {
    T dl = T(1.0) / qn.v;
    return chain(std::log(qn.v), dl, qn);
}; // natural log (qn.v > 0)

template <typename T, int NT>
inline qnumber_t<T, NT> tanh(
    const qnumber_t<T, NT>& qn
) {
    T tan = std::tanh(qn.v);
    T dtan = T(1.0) - T(std::pow(tan, 2.0));
    return chain(tan, dtan, qn);
}; // tanh function

template <typename T, int NT>
inline qnumber_t<T, NT> sin(
    const qnumber_t<T, NT>& qn
) {
    T sin = std::sin(qn.v);
    T dsin = std::cos(qn.v);
    return chain(sin, dsin, qn);
}; // sine function

template <typename T, int NT>
inline qnumber_t<T, NT> cos(
    const qnumber_t<T, NT>& qn
) {
    T cos = std::cos(qn.v);
    T dcos = -std::sin(qn.v);
    return chain(cos, dcos, qn);
}; // cosine function

template <typename T, int NT>
inline qnumber_t<T, NT> atan2(
    const qnumber_t<T, NT>& qn1,
    const qnumber_t<T, NT>& qn2
) {
    T v0 = std::atan2(qn1.v, qn2.v);
    T denom = qn1.v * qn1.v + qn2.v * qn2.v;
    T d1 = qn2.v / denom;
    T d2 = -qn1.v / denom;
    qnumber_t<T, NT> r(
        v0,
        d1 * qn1.g1 + d2 * qn2.g1,
        d1 * qn1.g2 + d2 * qn2.g2
    );
    if constexpr (NT == 4) {
        r.x1 = d1 * qn1.x1 + d2 * qn2.x1;
        r.x2 = d1 * qn1.x2 + d2 * qn2.x2;
    }
    return r;
}; // arctan2 function

template <typename T, int NT>
inline qnumber_t<T, NT> pow(
    const qnumber_t<T, NT>& qn,
    double n
) {
    T tmp0 = T(std::pow(qn.v, T(n - 1)));
    T tmp = T(n) * tmp0;
    return chain(tmp0 * qn.v, tmp, qn);
}; // power function

template <typename T, int NT>
inline qnumber_t<T, NT> sqrt(
    const qnumber_t<T, NT>& qn
) {
    T tmp0 = std::sqrt(qn.v);
    T tmp = T(0.5) / tmp0;
    return chain(tmp0, tmp, qn);
}; // sqrt function

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_QN_H
