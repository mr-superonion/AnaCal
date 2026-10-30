#ifndef ANACAL_MATH_NORM_H
#define ANACAL_MATH_NORM_H

#include "../stdafx.h"
#include "qnumber.h"
#include "tensor.h"

// Normalisation layers on qnumber tensors.  The statistics (mean,
// variance) are computed from the qnumber input, so their dependence on
// the shear and the position enters the responses; only the affine
// weights and, for BatchNorm2d in eval mode, the running statistics are
// constants.

namespace anacal {
namespace math {

namespace detail {
// Normalise G groups of n elements in place, (x - mean) / sqrt(var + eps)
// with the biased variance as in PyTorch.  Element i of group g sits at
// plane index g * stride_group + i * stride_elem; the loops run over the
// groups innermost, so a group stride of 1 (the per-pixel channel norm)
// vectorises.  Tangents: with m_k = mean of t_k and
// dvar_k = 2 mean((v - m)(t_k - m_k)),
//   y_k = (t_k - m_k) inv + (v - m) (-1/2) inv^3 dvar_k.
template <typename T, int NT>
inline void normalize_groups(
    qtensor_t<T, NT>& x, std::size_t G, std::size_t n,
    std::size_t stride_group, std::size_t stride_elem, double eps
) {
    constexpr int NC = qtensor_t<T, NT>::ncomp;
    std::vector<T> mean(G, T(0)), inv(G);
    std::vector<std::array<T, NC>> dstat(G);  // [0]: var, [k]: dvar_k
    for (auto& a : dstat) a.fill(T(0));
    T* v = x.plane(0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t g = 0; g < G; ++g) {
            mean[g] += v[g * stride_group + i * stride_elem];
        }
    }
    const T invn = T(1) / static_cast<T>(n);
    for (std::size_t g = 0; g < G; ++g) {
        mean[g] *= invn;
    }
    // tangent means
    std::vector<std::array<T, NC>> tmean(G);
    for (auto& a : tmean) a.fill(T(0));
    for (int c = 1; c < NC; ++c) {
        const T* t = x.plane(c);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t g = 0; g < G; ++g) {
                tmean[g][c] += t[g * stride_group + i * stride_elem];
            }
        }
        for (std::size_t g = 0; g < G; ++g) {
            tmean[g][c] *= invn;
        }
    }
    // variance and its tangents
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t g = 0; g < G; ++g) {
            const T d = v[g * stride_group + i * stride_elem] - mean[g];
            dstat[g][0] += d * d;
        }
    }
    for (int c = 1; c < NC; ++c) {
        const T* t = x.plane(c);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t g = 0; g < G; ++g) {
                const std::size_t idx = g * stride_group + i * stride_elem;
                dstat[g][c] += T(2) * (v[idx] - mean[g]) * (t[idx] - tmean[g][c]);
            }
        }
    }
    for (std::size_t g = 0; g < G; ++g) {
        const T var = dstat[g][0] * invn;
        inv[g] = T(1) / std::sqrt(var + T(eps));
        for (int c = 1; c < NC; ++c) {
            // d inv = -1/2 inv^3 dvar
            dstat[g][c] = T(-0.5) * inv[g] * inv[g] * inv[g] * (dstat[g][c] * invn);
        }
    }
    // apply: tangents first (they need the un-normalised value)
    for (int c = 1; c < NC; ++c) {
        T* t = x.plane(c);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t g = 0; g < G; ++g) {
                const std::size_t idx = g * stride_group + i * stride_elem;
                t[idx] = (t[idx] - tmean[g][c]) * inv[g] + (v[idx] - mean[g]) * dstat[g][c];
            }
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t g = 0; g < G; ++g) {
            const std::size_t idx = g * stride_group + i * stride_elem;
            v[idx] = (v[idx] - mean[g]) * inv[g];
        }
    }
}

// x = x * a + b with constants a, b on the elements plane index
// g * stride_group + i * stride_elem (i fixed): an affine map of one
// feature / channel i across the G groups.
template <typename T, int NT>
inline void affine_feature(
    qtensor_t<T, NT>& x, std::size_t G, std::size_t stride_group,
    std::size_t base, double a, double b
) {
    constexpr int NC = qtensor_t<T, NT>::ncomp;
    const T ta = static_cast<T>(a), tb = static_cast<T>(b);
    T* v = x.plane(0);
    for (std::size_t g = 0; g < G; ++g) {
        v[base + g * stride_group] = v[base + g * stride_group] * ta + tb;
    }
    for (int c = 1; c < NC; ++c) {
        T* t = x.plane(c);
        for (std::size_t g = 0; g < G; ++g) {
            t[base + g * stride_group] *= ta;
        }
    }
}
} // namespace detail

// torch.nn.LayerNorm(d) over the last dimension of [..., d]
struct LayerNorm {
    std::size_t features = 0;
    double eps = 1e-5;
    std::vector<double> weight;  // empty: elementwise_affine=False
    std::vector<double> bias;

    LayerNorm() = default;

    LayerNorm(
        std::size_t features,
        std::vector<double> weight = {},
        std::vector<double> bias = {},
        double eps = 1e-5
    )
        : features(features), eps(eps), weight(std::move(weight)),
          bias(std::move(bias)) {
        if (!this->weight.empty() && this->weight.size() != features) {
            throw std::invalid_argument("LayerNorm: weight size mismatch");
        }
        if (!this->bias.empty() && this->bias.size() != features) {
            throw std::invalid_argument("LayerNorm: bias size mismatch");
        }
    }

    template <typename T, int NT>
    qtensor_t<T, NT> forward(const qtensor_t<T, NT>& x) const {
        if (x.ndim() == 0 || x.shape().back() != features) {
            throw std::invalid_argument(
                "LayerNorm: last dimension must be features"
            );
        }
        qtensor_t<T, NT> out = x.copy();
        const std::size_t rows = out.size() / features;
        detail::normalize_groups(out, rows, features, features, 1, eps);
        if (!weight.empty() || !bias.empty()) {
            for (std::size_t i = 0; i < features; ++i) {
                detail::affine_feature(
                    out, rows, features, i,
                    weight.empty() ? 1.0 : weight[i], bias.empty() ? 0.0 : bias[i]
                );
            }
        }
        return out;
    }
};

// Layer norm over the channels of [C, H, W] at each pixel (the
// ChannelLayerNorm2d of the D4 CNN: GroupNorm-free, no coupling between
// pixels), with per-channel affine weights
struct ChannelLayerNorm2d {
    std::size_t channels = 0;
    double eps = 1e-5;
    std::vector<double> weight;  // empty: no affine
    std::vector<double> bias;

    ChannelLayerNorm2d() = default;

    ChannelLayerNorm2d(
        std::size_t channels,
        std::vector<double> weight = {},
        std::vector<double> bias = {},
        double eps = 1e-5
    )
        : channels(channels), eps(eps), weight(std::move(weight)),
          bias(std::move(bias)) {
        if (!this->weight.empty() && this->weight.size() != channels) {
            throw std::invalid_argument(
                "ChannelLayerNorm2d: weight size mismatch"
            );
        }
        if (!this->bias.empty() && this->bias.size() != channels) {
            throw std::invalid_argument(
                "ChannelLayerNorm2d: bias size mismatch"
            );
        }
    }

    template <typename T, int NT>
    qtensor_t<T, NT> forward(const qtensor_t<T, NT>& x) const {
        if (x.ndim() != 3 || x.shape()[0] != channels) {
            throw std::invalid_argument(
                "ChannelLayerNorm2d: input must be [channels, H, W]"
            );
        }
        qtensor_t<T, NT> out = x.copy();
        const std::size_t npix = x.shape()[1] * x.shape()[2];
        detail::normalize_groups(out, npix, channels, 1, npix, eps);
        if (!weight.empty() || !bias.empty()) {
            for (std::size_t c = 0; c < channels; ++c) {
                detail::affine_feature(
                    out, npix, 1, c * npix,
                    weight.empty() ? 1.0 : weight[c], bias.empty() ? 0.0 : bias[c]
                );
            }
        }
        return out;
    }
};

// torch.nn.BatchNorm2d in eval mode on [C, H, W]: a per-channel affine
// map with the running statistics
struct BatchNorm2d {
    std::size_t channels = 0;
    std::vector<double> scale;  // weight / sqrt(running_var + eps)
    std::vector<double> shift;  // bias - running_mean * scale

    BatchNorm2d() = default;

    BatchNorm2d(
        std::size_t channels,
        const std::vector<double>& running_mean,
        const std::vector<double>& running_var,
        std::vector<double> weight = {},
        std::vector<double> bias = {},
        double eps = 1e-5
    )
        : channels(channels), scale(channels), shift(channels) {
        if (running_mean.size() != channels
            || running_var.size() != channels
            || (!weight.empty() && weight.size() != channels)
            || (!bias.empty() && bias.size() != channels)) {
            throw std::invalid_argument("BatchNorm2d: size mismatch");
        }
        for (std::size_t c = 0; c < channels; ++c) {
            const double a = weight.empty() ? 1.0 : weight[c];
            const double b = bias.empty() ? 0.0 : bias[c];
            scale[c] = a / std::sqrt(running_var[c] + eps);
            shift[c] = b - running_mean[c] * scale[c];
        }
    }

    template <typename T, int NT>
    qtensor_t<T, NT> forward(const qtensor_t<T, NT>& x) const {
        if (x.ndim() != 3 || x.shape()[0] != channels) {
            throw std::invalid_argument(
                "BatchNorm2d: input must be [channels, H, W]"
            );
        }
        qtensor_t<T, NT> out = x.copy();
        const std::size_t npix = x.shape()[1] * x.shape()[2];
        for (std::size_t c = 0; c < channels; ++c) {
            detail::affine_feature(out, npix, 1, c * npix, scale[c], shift[c]);
        }
        return out;
    }
};

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_NORM_H
