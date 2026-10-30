#ifndef ANACAL_MATH_ACTIVATION_H
#define ANACAL_MATH_ACTIVATION_H

#include "../stdafx.h"
#include "qnumber.h"
#include "tensor.h"

// Activation functions of a network evaluated on qnumbers: the value is
// the PyTorch activation, and the g1, g2, x1, x2 slots carry its chain
// rule, so a network built from these layers returns its output together
// with the shear and position responses of the input image.  At the kinks
// of relu, leaky_relu and max pooling the derivative is the one PyTorch's
// autograd uses (relu'(0) = 0, leaky_relu'(0) = slope).

namespace anacal {
namespace math {

// f(x) with derivative df at x.v applied to all four tangent slots.
inline qnumber chain(double f, double df, const qnumber& x) {
    return qnumber(f, df * x.g1, df * x.g2, df * x.x1, df * x.x2);
}

inline qnumber relu(const qnumber& x) {
    return (x.v > 0.0) ? x : qnumber();
}

inline qnumber leaky_relu(const qnumber& x, double slope = 0.01) {
    return (x.v > 0.0) ? x : slope * x;
}

inline double sigmoid(double x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

inline qnumber sigmoid(const qnumber& x) {
    const double s = sigmoid(x.v);
    return chain(s, s * (1.0 - s), x);
}

// x * sigmoid(x) (torch.nn.SiLU)
inline qnumber silu(const qnumber& x) {
    const double s = sigmoid(x.v);
    return chain(x.v * s, s + x.v * s * (1.0 - s), x);
}

// torch.nn.GELU: x Phi(x) with the exact erf (approximate="none") or the
// tanh approximation (approximate="tanh")
inline qnumber gelu(const qnumber& x, bool tanh_approx = false) {
    const double v = x.v;
    if (tanh_approx) {
        constexpr double k0 = 0.7978845608028654;  // sqrt(2 / pi)
        constexpr double k1 = 0.044715;
        const double t = std::tanh(k0 * (v + k1 * v * v * v));
        const double dt = (1.0 - t * t) * k0 * (1.0 + 3.0 * k1 * v * v);
        return chain(0.5 * v * (1.0 + t), 0.5 * (1.0 + t) + 0.5 * v * dt, x);
    }
    const double cdf = 0.5 * std::erfc(-v * M_SQRT1_2);
    const double pdf = std::exp(-0.5 * v * v) * 0.3989422804014327;
    return chain(v * cdf, cdf + v * pdf, x);
}

// torch.nn.Softplus: log(1 + exp(beta x)) / beta, linear where
// beta x > threshold
inline qnumber softplus(
    const qnumber& x, double beta = 1.0, double threshold = 20.0
) {
    const double bx = beta * x.v;
    if (bx > threshold) {
        return x;
    }
    return chain(std::log1p(std::exp(bx)) / beta, sigmoid(bx), x);
}

// Element-wise versions on tensors
inline qtensor relu(const qtensor& x) {
    return elementwise(x, [](const qnumber& q) { return relu(q); });
}

inline qtensor leaky_relu(const qtensor& x, double slope = 0.01) {
    return elementwise(
        x, [slope](const qnumber& q) { return leaky_relu(q, slope); }
    );
}

inline qtensor sigmoid(const qtensor& x) {
    return elementwise(x, [](const qnumber& q) { return sigmoid(q); });
}

inline qtensor silu(const qtensor& x) {
    return elementwise(x, [](const qnumber& q) { return silu(q); });
}

inline qtensor tanh(const qtensor& x) {
    return elementwise(x, [](const qnumber& q) { return tanh(q); });
}

inline qtensor gelu(const qtensor& x, bool tanh_approx = false) {
    return elementwise(
        x, [tanh_approx](const qnumber& q) { return gelu(q, tanh_approx); }
    );
}

inline qtensor softplus(
    const qtensor& x, double beta = 1.0, double threshold = 20.0
) {
    return elementwise(x, [beta, threshold](const qnumber& q) {
        return softplus(q, beta, threshold);
    });
}

// torch.softmax over the last dimension
inline qtensor softmax(const qtensor& x) {
    if (x.ndim() == 0) {
        throw std::invalid_argument("softmax: empty shape");
    }
    qtensor out = qtensor::from_vector(x.to_vector(), x.shape());
    const std::size_t d = out.shape().back();
    const std::size_t rows = (d == 0) ? 0 : out.size() / d;
    qnumber* p = out.data();
    for (std::size_t r = 0; r < rows; ++r) {
        qnumber* row = p + r * d;
        double vmax = row[0].v;
        for (std::size_t i = 1; i < d; ++i) {
            vmax = std::max(vmax, row[i].v);
        }
        qnumber sum;
        for (std::size_t i = 0; i < d; ++i) {
            row[i] = exp(row[i] - vmax);
            sum = sum + row[i];
        }
        for (std::size_t i = 0; i < d; ++i) {
            row[i] = row[i] / sum;
        }
    }
    return out;
}

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_ACTIVATION_H
