#ifndef ANACAL_MATH_ACTIVATION_H
#define ANACAL_MATH_ACTIVATION_H

#include "../stdafx.h"
#include "qnumber.h"
#include "tensor.h"

// Activation functions of a network evaluated on qnumbers: the value is
// the PyTorch activation, and the tangent slots carry its chain rule, so
// a network built from these layers returns its output together with the
// shear (and position) responses of the input image.  At the kinks of
// relu, leaky_relu and max pooling the derivative is the one PyTorch's
// autograd uses (relu'(0) = 0, leaky_relu'(0) = slope).  The tensor
// versions run plane by plane (tensor.h, elementwise_chain).

namespace anacal {
namespace math {

namespace act {
// value and derivative of each activation at v
template <typename T>
inline void relu(T v, T& f, T& df) {
    const bool on = v > T(0);
    f = on ? v : T(0);
    df = on ? T(1) : T(0);
}

template <typename T>
inline void leaky_relu(T v, double slope, T& f, T& df) {
    const bool on = v > T(0);
    f = on ? v : T(slope) * v;
    df = on ? T(1) : T(slope);
}

template <typename T>
inline T sigmoid(T x) {
    if (x >= T(0)) {
        return T(1) / (T(1) + std::exp(-x));
    }
    const T e = std::exp(x);
    return e / (T(1) + e);
}

template <typename T>
inline void sigmoid(T v, T& f, T& df) {
    const T s = sigmoid(v);
    f = s;
    df = s * (T(1) - s);
}

// x * sigmoid(x) (torch.nn.SiLU)
template <typename T>
inline void silu(T v, T& f, T& df) {
    const T s = sigmoid(v);
    f = v * s;
    df = s + v * s * (T(1) - s);
}

template <typename T>
inline void tanh(T v, T& f, T& df) {
    const T t = std::tanh(v);
    f = t;
    df = T(1) - t * t;
}

// torch.nn.GELU: x Phi(x) with the exact erf (approximate="none") or the
// tanh approximation (approximate="tanh")
template <typename T>
inline void gelu(T v, bool tanh_approx, T& f, T& df) {
    if (tanh_approx) {
        constexpr T k0 = T(0.7978845608028654);  // sqrt(2 / pi)
        constexpr T k1 = T(0.044715);
        const T t = std::tanh(k0 * (v + k1 * v * v * v));
        const T dt = (T(1) - t * t) * k0 * (T(1) + T(3) * k1 * v * v);
        f = T(0.5) * v * (T(1) + t);
        df = T(0.5) * (T(1) + t) + T(0.5) * v * dt;
        return;
    }
    const T cdf = T(0.5) * std::erfc(-v * T(M_SQRT1_2));
    const T pdf = std::exp(T(-0.5) * v * v) * T(0.3989422804014327);
    f = v * cdf;
    df = cdf + v * pdf;
}

// torch.nn.Softplus: log(1 + exp(beta x)) / beta, linear where
// beta x > threshold
template <typename T>
inline void softplus(T v, double beta, double threshold, T& f, T& df) {
    const T bx = T(beta) * v;
    if (bx > T(threshold)) {
        f = v;
        df = T(1);
        return;
    }
    f = std::log1p(std::exp(bx)) / T(beta);
    df = sigmoid(bx);
}
} // namespace act

// qnumber versions
template <typename T, int NT>
inline qnumber_t<T, NT> relu(const qnumber_t<T, NT>& x) {
    T f, df;
    act::relu(x.v, f, df);
    return chain(f, df, x);
}

template <typename T, int NT>
inline qnumber_t<T, NT> leaky_relu(const qnumber_t<T, NT>& x, double slope = 0.01) {
    T f, df;
    act::leaky_relu(x.v, slope, f, df);
    return chain(f, df, x);
}

template <typename T, int NT>
inline qnumber_t<T, NT> sigmoid(const qnumber_t<T, NT>& x) {
    T f, df;
    act::sigmoid(x.v, f, df);
    return chain(f, df, x);
}

template <typename T, int NT>
inline qnumber_t<T, NT> silu(const qnumber_t<T, NT>& x) {
    T f, df;
    act::silu(x.v, f, df);
    return chain(f, df, x);
}

template <typename T, int NT>
inline qnumber_t<T, NT> gelu(const qnumber_t<T, NT>& x, bool tanh_approx = false) {
    T f, df;
    act::gelu(x.v, tanh_approx, f, df);
    return chain(f, df, x);
}

template <typename T, int NT>
inline qnumber_t<T, NT> softplus(
    const qnumber_t<T, NT>& x, double beta = 1.0, double threshold = 20.0
) {
    T f, df;
    act::softplus(x.v, beta, threshold, f, df);
    return chain(f, df, x);
}

// Element-wise versions on tensors
template <typename T, int NT>
inline qtensor_t<T, NT> relu(const qtensor_t<T, NT>& x) {
    return elementwise_chain(x, [](T v, T& f, T& df) { act::relu(v, f, df); });
}

template <typename T, int NT>
inline qtensor_t<T, NT> leaky_relu(const qtensor_t<T, NT>& x, double slope = 0.01) {
    return elementwise_chain(
        x, [slope](T v, T& f, T& df) { act::leaky_relu(v, slope, f, df); }
    );
}

template <typename T, int NT>
inline qtensor_t<T, NT> sigmoid(const qtensor_t<T, NT>& x) {
    return elementwise_chain(x, [](T v, T& f, T& df) { act::sigmoid(v, f, df); });
}

template <typename T, int NT>
inline qtensor_t<T, NT> silu(const qtensor_t<T, NT>& x) {
    return elementwise_chain(x, [](T v, T& f, T& df) { act::silu(v, f, df); });
}

template <typename T, int NT>
inline qtensor_t<T, NT> tanh(const qtensor_t<T, NT>& x) {
    return elementwise_chain(x, [](T v, T& f, T& df) { act::tanh(v, f, df); });
}

namespace act {
// exp(x) for x <= 0 from a degree-6 polynomial on [-ln2/2, ln2/2] and the
// exponent bits: relative error ~1e-7, every step vectorises (the float
// GELU below is the one transcendental loop of the CNN).
inline float fast_exp_neg(float x) {
    x = std::max(x, -87.0f);
    const float n = std::floor(x * 1.44269504f + 0.5f);
    const float r = x - n * 0.69314718f;
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    const int e = (static_cast<int>(n) + 127) << 23;
    float scale;
    __builtin_memcpy(&scale, &e, 4);
    return p * scale;
}

// Exact-erf GELU in float through Abramowitz-Stegun 7.1.26 (|erf error|
// < 1.5e-7): erfc(|x|) = P(t) exp(-x^2), t = 1 / (1 + 0.3275911 |x|),
// shares the exponential with the Gaussian pdf of the derivative.
inline void gelu_fast(float v, float& f, float& df) {
    const float x = std::fabs(v) * 0.70710678f;
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float poly = t * (0.254829592f + t * (-0.284496736f
        + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    const float e = fast_exp_neg(-x * x);
    const float erfc_abs = poly * e;
    const float cdf = (v >= 0.0f) ? 1.0f - 0.5f * erfc_abs : 0.5f * erfc_abs;
    const float pdf = e * 0.39894228f;
    f = v * cdf;
    df = cdf + v * pdf;
}
} // namespace act

template <typename T, int NT>
inline qtensor_t<T, NT> gelu(const qtensor_t<T, NT>& x, bool tanh_approx = false) {
    if constexpr (std::is_same<T, float>::value) {
        if (!tanh_approx) {
            return elementwise_chain(
                x, [](float v, float& f, float& df) { act::gelu_fast(v, f, df); }
            );
        }
    }
    return elementwise_chain(
        x, [tanh_approx](T v, T& f, T& df) { act::gelu(v, tanh_approx, f, df); }
    );
}

template <typename T, int NT>
inline qtensor_t<T, NT> softplus(
    const qtensor_t<T, NT>& x, double beta = 1.0, double threshold = 20.0
) {
    return elementwise_chain(x, [beta, threshold](T v, T& f, T& df) {
        act::softplus(v, beta, threshold, f, df);
    });
}

// torch.softmax over the last dimension: s_i = e_i / sum e, with the
// tangent ds_i = s_i (t_i - sum_j s_j t_j)
template <typename T, int NT>
inline qtensor_t<T, NT> softmax(const qtensor_t<T, NT>& x) {
    if (x.ndim() == 0) {
        throw std::invalid_argument("softmax: empty shape");
    }
    const qtensor_t<T, NT> xc = contiguous(x);
    qtensor_t<T, NT> out(xc.shape());
    const std::size_t d = xc.shape().back();
    const std::size_t rows = (d == 0) ? 0 : xc.size() / d;
    const T* v = xc.plane(0);
    T* ov = out.plane(0);
    for (std::size_t r = 0; r < rows; ++r) {
        const T* vr = v + r * d;
        T* sr = ov + r * d;
        T vmax = vr[0];
        for (std::size_t i = 1; i < d; ++i) {
            vmax = std::max(vmax, vr[i]);
        }
        T sum = 0;
        for (std::size_t i = 0; i < d; ++i) {
            sr[i] = std::exp(vr[i] - vmax);
            sum += sr[i];
        }
        for (std::size_t i = 0; i < d; ++i) {
            sr[i] /= sum;
        }
        for (int c = 1; c < qtensor_t<T, NT>::ncomp; ++c) {
            const T* tr = xc.plane(c) + r * d;
            T* o = out.plane(c) + r * d;
            T dot = 0;
            for (std::size_t i = 0; i < d; ++i) {
                dot += sr[i] * tr[i];
            }
            for (std::size_t i = 0; i < d; ++i) {
                o[i] = sr[i] * (tr[i] - dot);
            }
        }
    }
    return out;
}

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_ACTIVATION_H
