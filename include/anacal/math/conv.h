#ifndef ANACAL_MATH_CONV_H
#define ANACAL_MATH_CONV_H

#include "../stdafx.h"
#include "qnumber.h"
#include "tensor.h"

// Linear layers of a network on qnumber tensors.  The weights are plain
// doubles in PyTorch's layout (they do not depend on the image), so each
// layer is linear in its qnumber input and the four tangent slots go
// through the same weights as the value.

namespace anacal {
namespace math {

// torch.nn.Conv2d on one sample: input [C_in, H, W], output
// [C_out, H_out, W_out], zero padding, weight [C_out, C_in, kh, kw]
struct Conv2d {
    std::size_t in_channels = 0;
    std::size_t out_channels = 0;
    std::size_t kh = 0;
    std::size_t kw = 0;
    std::size_t stride = 1;
    std::size_t padding = 0;
    std::size_t dilation = 1;
    std::vector<double> weight;
    std::vector<double> bias;  // empty: no bias

    Conv2d() = default;

    Conv2d(
        std::size_t in_channels,
        std::size_t out_channels,
        std::size_t kh,
        std::size_t kw,
        std::vector<double> weight,
        std::vector<double> bias = {},
        std::size_t stride = 1,
        std::size_t padding = 0,
        std::size_t dilation = 1
    )
        : in_channels(in_channels), out_channels(out_channels),
          kh(kh), kw(kw), stride(stride), padding(padding),
          dilation(dilation), weight(std::move(weight)),
          bias(std::move(bias)) {
        if (this->weight.size() != out_channels * in_channels * kh * kw) {
            throw std::invalid_argument("Conv2d: weight size mismatch");
        }
        if (!this->bias.empty() && this->bias.size() != out_channels) {
            throw std::invalid_argument("Conv2d: bias size mismatch");
        }
        if (stride == 0 || dilation == 0 || kh == 0 || kw == 0) {
            throw std::invalid_argument("Conv2d: zero stride/dilation/kernel");
        }
    }

    qtensor forward(const qtensor& x) const {
        if (x.ndim() != 3 || x.shape()[0] != in_channels) {
            throw std::invalid_argument(
                "Conv2d: input must be [in_channels, H, W]"
            );
        }
        const qtensor xc = contiguous(x);
        const std::size_t h = xc.shape()[1];
        const std::size_t w = xc.shape()[2];
        const std::size_t ekh = dilation * (kh - 1) + 1;
        const std::size_t ekw = dilation * (kw - 1) + 1;
        if (h + 2 * padding < ekh || w + 2 * padding < ekw) {
            throw std::invalid_argument("Conv2d: kernel larger than input");
        }
        const std::size_t ho = (h + 2 * padding - ekh) / stride + 1;
        const std::size_t wo = (w + 2 * padding - ekw) / stride + 1;
        qtensor out({out_channels, ho, wo});
        const qnumber* src = xc.data();
        qnumber* dst = out.data();
        const long pad = static_cast<long>(padding);
        for (std::size_t co = 0; co < out_channels; ++co) {
            qnumber* plane = dst + co * ho * wo;
            if (!bias.empty()) {
                for (std::size_t i = 0; i < ho * wo; ++i) {
                    plane[i].v = bias[co];
                }
            }
            for (std::size_t ci = 0; ci < in_channels; ++ci) {
                const qnumber* in = src + ci * h * w;
                const double* wk = weight.data()
                    + (co * in_channels + ci) * kh * kw;
                for (std::size_t ky = 0; ky < kh; ++ky) {
                    for (std::size_t kx = 0; kx < kw; ++kx) {
                        const double a = wk[ky * kw + kx];
                        if (a == 0.0) {
                            continue;
                        }
                        const long dy = static_cast<long>(ky * dilation) - pad;
                        const long dx = static_cast<long>(kx * dilation) - pad;
                        for (std::size_t oy = 0; oy < ho; ++oy) {
                            const long iy = static_cast<long>(oy * stride) + dy;
                            if (iy < 0 || iy >= static_cast<long>(h)) {
                                continue;
                            }
                            const qnumber* row = in + iy * w;
                            qnumber* orow = plane + oy * wo;
                            for (std::size_t ox = 0; ox < wo; ++ox) {
                                const long ix =
                                    static_cast<long>(ox * stride) + dx;
                                if (ix < 0 || ix >= static_cast<long>(w)) {
                                    continue;
                                }
                                axpy(orow[ox], a, row[ix]);
                            }
                        }
                    }
                }
            }
        }
        return out;
    }
};

// torch.nn.Linear over the last dimension: input [..., in_features],
// output [..., out_features], weight [out_features, in_features]
struct Linear {
    std::size_t in_features = 0;
    std::size_t out_features = 0;
    std::vector<double> weight;
    std::vector<double> bias;  // empty: no bias

    Linear() = default;

    Linear(
        std::size_t in_features,
        std::size_t out_features,
        std::vector<double> weight,
        std::vector<double> bias = {}
    )
        : in_features(in_features), out_features(out_features),
          weight(std::move(weight)), bias(std::move(bias)) {
        if (this->weight.size() != out_features * in_features) {
            throw std::invalid_argument("Linear: weight size mismatch");
        }
        if (!this->bias.empty() && this->bias.size() != out_features) {
            throw std::invalid_argument("Linear: bias size mismatch");
        }
    }

    qtensor forward(const qtensor& x) const {
        if (x.ndim() == 0 || x.shape().back() != in_features) {
            throw std::invalid_argument(
                "Linear: last dimension must be in_features"
            );
        }
        const qtensor xc = contiguous(x);
        std::vector<std::size_t> shape = xc.shape();
        const std::size_t rows = xc.size() / in_features;
        shape.back() = out_features;
        qtensor out(shape);
        const qnumber* src = xc.data();
        qnumber* dst = out.data();
        for (std::size_t r = 0; r < rows; ++r) {
            const qnumber* in = src + r * in_features;
            qnumber* o = dst + r * out_features;
            for (std::size_t j = 0; j < out_features; ++j) {
                qnumber acc(bias.empty() ? 0.0 : bias[j]);
                const double* wj = weight.data() + j * in_features;
                for (std::size_t i = 0; i < in_features; ++i) {
                    axpy(acc, wj[i], in[i]);
                }
                o[j] = acc;
            }
        }
        return out;
    }
};

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_CONV_H
