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
// Normalise n values spaced by stride in place: (x - mean) / sqrt(var +
// eps), with the biased variance as in PyTorch.
inline void normalize_strided(
    qnumber* x, std::size_t n, std::size_t stride, double eps
) {
    qnumber mean;
    for (std::size_t i = 0; i < n; ++i) {
        mean = mean + x[i * stride];
    }
    mean = mean / static_cast<double>(n);
    qnumber var;
    for (std::size_t i = 0; i < n; ++i) {
        const qnumber d = x[i * stride] - mean;
        var = var + d * d;
    }
    var = var / static_cast<double>(n);
    const qnumber inv = 1.0 / sqrt(var + eps);
    for (std::size_t i = 0; i < n; ++i) {
        x[i * stride] = (x[i * stride] - mean) * inv;
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

    qtensor forward(const qtensor& x) const {
        if (x.ndim() == 0 || x.shape().back() != features) {
            throw std::invalid_argument(
                "LayerNorm: last dimension must be features"
            );
        }
        qtensor out = qtensor::from_vector(x.to_vector(), x.shape());
        qnumber* p = out.data();
        const std::size_t rows = out.size() / features;
        for (std::size_t r = 0; r < rows; ++r) {
            qnumber* row = p + r * features;
            detail::normalize_strided(row, features, 1, eps);
            for (std::size_t i = 0; i < features; ++i) {
                if (!weight.empty()) {
                    row[i] = row[i] * weight[i];
                }
                if (!bias.empty()) {
                    row[i] = row[i] + bias[i];
                }
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

    qtensor forward(const qtensor& x) const {
        if (x.ndim() != 3 || x.shape()[0] != channels) {
            throw std::invalid_argument(
                "ChannelLayerNorm2d: input must be [channels, H, W]"
            );
        }
        qtensor out = qtensor::from_vector(x.to_vector(), x.shape());
        qnumber* p = out.data();
        const std::size_t npix = x.shape()[1] * x.shape()[2];
        for (std::size_t i = 0; i < npix; ++i) {
            detail::normalize_strided(p + i, channels, npix, eps);
        }
        if (!weight.empty() || !bias.empty()) {
            for (std::size_t c = 0; c < channels; ++c) {
                qnumber* plane = p + c * npix;
                const double a = weight.empty() ? 1.0 : weight[c];
                const double b = bias.empty() ? 0.0 : bias[c];
                for (std::size_t i = 0; i < npix; ++i) {
                    plane[i] = plane[i] * a + b;
                }
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

    qtensor forward(const qtensor& x) const {
        if (x.ndim() != 3 || x.shape()[0] != channels) {
            throw std::invalid_argument(
                "BatchNorm2d: input must be [channels, H, W]"
            );
        }
        qtensor out = qtensor::from_vector(x.to_vector(), x.shape());
        qnumber* p = out.data();
        const std::size_t npix = x.shape()[1] * x.shape()[2];
        for (std::size_t c = 0; c < channels; ++c) {
            qnumber* plane = p + c * npix;
            for (std::size_t i = 0; i < npix; ++i) {
                plane[i] = plane[i] * scale[c] + shift[c];
            }
        }
        return out;
    }
};

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_NORM_H
