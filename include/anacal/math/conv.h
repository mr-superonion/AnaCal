#ifndef ANACAL_MATH_CONV_H
#define ANACAL_MATH_CONV_H

#include "../stdafx.h"
#include "qnumber.h"
#include "tensor.h"

// Linear layers of a network on qnumber tensors.  The weights are plain
// doubles in PyTorch's layout (they do not depend on the image), so each
// layer is linear in its qnumber input and the tangent planes go through
// the same weights as the value plane.

namespace anacal {
namespace math {

namespace conv_detail {

constexpr std::size_t CO_BLK = 4;  // output channels per register tile
constexpr std::size_t PX = 8;      // pixels per register tile

// One block of CO_BLK output channels of a stride-1 convolution on the
// zero-padded component planes inp [C_in][nc][hp][wpad] with the packed
// weights wpk [C_in][kh][kw][CO_BLK], into outp [CO_BLK][nc][ho][wopad].
// A register tile of CO_BLK channels x PX pixels is accumulated over
// (C_in, kh, kw), so every input load feeds CO_BLK multiply-adds.
//
// Two bodies: ``simd`` uses 32-byte GCC vector extensions and is compiled
// for AVX2 + FMA; ``generic`` is the plain-array version for any target.
// Both run on every CPU in principle, but the vector-extension body is
// only fast with AVX2, so block_soa picks it at run time.
#define ANACAL_CONV_BLOCK_SIMD_BODY                                          \
    template <typename T>                                                    \
    void block_soa(                                                          \
        const T* inp, std::size_t in_channels, std::size_t hp,               \
        std::size_t wpad, const T* wpk, std::size_t kh, std::size_t kw,      \
        T* outp, std::size_t ho, std::size_t wopad, std::size_t nc           \
    ) {                                                                      \
        typedef T vec __attribute__((vector_size(32)));                      \
        constexpr std::size_t L = 32 / sizeof(T);                            \
        constexpr std::size_t NV = PX / L;                                   \
        const std::size_t plane_in = hp * wpad;                              \
        const std::size_t plane_out = ho * wopad;                            \
        for (std::size_t comp = 0; comp < nc; ++comp) {                      \
            for (std::size_t oy = 0; oy < ho; ++oy) {                        \
                for (std::size_t ox = 0; ox < wopad; ox += PX) {             \
                    vec acc[CO_BLK][NV];                                     \
                    for (std::size_t a = 0; a < CO_BLK; ++a) {               \
                        for (std::size_t n = 0; n < NV; ++n) {               \
                            acc[a][n] = vec{};                               \
                        }                                                    \
                    }                                                        \
                    const T* wk = wpk;                                       \
                    for (std::size_t ci = 0; ci < in_channels; ++ci) {       \
                        const T* ibase = inp + (ci * nc + comp) * plane_in   \
                            + oy * wpad + ox;                                \
                        for (std::size_t ky = 0; ky < kh; ++ky) {            \
                            const T* irow = ibase + ky * wpad;               \
                            for (std::size_t kx = 0; kx < kw; ++kx) {        \
                                const T* ip = irow + kx;                     \
                                vec xv[NV];                                  \
                                for (std::size_t n = 0; n < NV; ++n) {       \
                                    __builtin_memcpy(&xv[n], ip + n * L, 32); \
                                }                                            \
                                for (std::size_t a = 0; a < CO_BLK; ++a) {   \
                                    const vec wv = vec{} + wk[a];            \
                                    for (std::size_t n = 0; n < NV; ++n) {   \
                                        acc[a][n] += wv * xv[n];             \
                                    }                                        \
                                }                                            \
                                wk += CO_BLK;                                \
                            }                                                \
                        }                                                    \
                    }                                                        \
                    for (std::size_t a = 0; a < CO_BLK; ++a) {               \
                        T* o = outp + (a * nc + comp) * plane_out            \
                            + oy * wopad + ox;                               \
                        for (std::size_t n = 0; n < NV; ++n) {               \
                            __builtin_memcpy(o + n * L, &acc[a][n], 32);     \
                        }                                                    \
                    }                                                        \
                }                                                            \
            }                                                                \
        }                                                                    \
    }

namespace generic {
template <typename T>
void block_soa(
    const T* inp, std::size_t in_channels, std::size_t hp,
    std::size_t wpad, const T* wpk, std::size_t kh, std::size_t kw,
    T* outp, std::size_t ho, std::size_t wopad, std::size_t nc
) {
    const std::size_t plane_in = hp * wpad;
    const std::size_t plane_out = ho * wopad;
    for (std::size_t comp = 0; comp < nc; ++comp) {
        for (std::size_t oy = 0; oy < ho; ++oy) {
            for (std::size_t ox = 0; ox < wopad; ox += PX) {
                T acc[CO_BLK][PX];
                for (std::size_t a = 0; a < CO_BLK; ++a) {
                    for (std::size_t j = 0; j < PX; ++j) {
                        acc[a][j] = T(0);
                    }
                }
                const T* wk = wpk;
                for (std::size_t ci = 0; ci < in_channels; ++ci) {
                    const T* ibase = inp + (ci * nc + comp) * plane_in
                        + oy * wpad + ox;
                    for (std::size_t ky = 0; ky < kh; ++ky) {
                        const T* irow = ibase + ky * wpad;
                        for (std::size_t kx = 0; kx < kw; ++kx) {
                            const T* ip = irow + kx;
                            for (std::size_t a = 0; a < CO_BLK; ++a) {
                                const T wv = wk[a];
                                for (std::size_t j = 0; j < PX; ++j) {
                                    acc[a][j] += wv * ip[j];
                                }
                            }
                            wk += CO_BLK;
                        }
                    }
                }
                for (std::size_t a = 0; a < CO_BLK; ++a) {
                    T* o = outp + (a * nc + comp) * plane_out + oy * wopad + ox;
                    for (std::size_t j = 0; j < PX; ++j) {
                        o[j] = acc[a][j];
                    }
                }
            }
        }
    }
}
} // namespace generic

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define ANACAL_CONV_HAVE_AVX2_BODY 1
#pragma GCC push_options
#pragma GCC target("avx2,fma")
namespace avx2 {
ANACAL_CONV_BLOCK_SIMD_BODY
} // namespace avx2
#pragma GCC pop_options
#else
#define ANACAL_CONV_HAVE_AVX2_BODY 0
#endif
#undef ANACAL_CONV_BLOCK_SIMD_BODY

inline bool cpu_has_avx2_fma() {
#if ANACAL_CONV_HAVE_AVX2_BODY
    static const bool ok = __builtin_cpu_supports("avx2")
        && __builtin_cpu_supports("fma");
    return ok;
#else
    return false;
#endif
}

template <typename T>
inline void block_soa(
    const T* inp, std::size_t in_channels, std::size_t hp,
    std::size_t wpad, const T* wpk, std::size_t kh, std::size_t kw,
    T* outp, std::size_t ho, std::size_t wopad, std::size_t nc
) {
#if ANACAL_CONV_HAVE_AVX2_BODY
    if (cpu_has_avx2_fma()) {
        avx2::block_soa<T>(
            inp, in_channels, hp, wpad, wpk, kh, kw, outp, ho, wopad, nc
        );
        return;
    }
#endif
    generic::block_soa<T>(
        inp, in_channels, hp, wpad, wpk, kh, kw, outp, ho, wopad, nc
    );
}

} // namespace conv_detail

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

    // Stride 1 and dilation 1 (every layer of the shape CNNs) take the
    // blocked kernel; anything else the plain loop.
    template <typename T, int NT>
    qtensor_t<T, NT> forward(const qtensor_t<T, NT>& x) const {
        if (x.ndim() != 3 || x.shape()[0] != in_channels) {
            throw std::invalid_argument(
                "Conv2d: input must be [in_channels, H, W]"
            );
        }
        const qtensor_t<T, NT> xc = contiguous(x);
        const std::size_t h = xc.shape()[1];
        const std::size_t w = xc.shape()[2];
        const std::size_t ekh = dilation * (kh - 1) + 1;
        const std::size_t ekw = dilation * (kw - 1) + 1;
        if (h + 2 * padding < ekh || w + 2 * padding < ekw) {
            throw std::invalid_argument("Conv2d: kernel larger than input");
        }
        if (stride == 1 && dilation == 1) {
            return forward_blocked(xc, h, w);
        }
        return forward_generic(xc, h, w, ekh, ekw);
    }

    // The component planes are independent linear channels of the same
    // convolution.  They are copied once into one zero-padded buffer
    // (padding rows / columns, plus a tail so the last register tile
    // reads zeros), the kernel runs per block of CO_BLK output channels
    // (padded with zero weights), and the valid part is copied into the
    // output planes with the bias added to the value plane.
    template <typename T, int NT>
    qtensor_t<T, NT> forward_blocked(
        const qtensor_t<T, NT>& xc, std::size_t h, std::size_t w
    ) const {
        using conv_detail::CO_BLK;
        using conv_detail::PX;
        constexpr std::size_t nc = qtensor_t<T, NT>::ncomp;
        const std::size_t hp = h + 2 * padding;
        const std::size_t wp = w + 2 * padding;
        const std::size_t ho = hp - kh + 1;
        const std::size_t wo = wp - kw + 1;
        const std::size_t wopad = (wo + PX - 1) / PX * PX;
        const std::size_t wpad = wopad + kw - 1;  // >= wp: tail reads are zeros
        const std::size_t plane_in = hp * wpad;
        const std::size_t plane_out = ho * wopad;

        std::vector<T> inp(in_channels * nc * plane_in, T(0));
        for (std::size_t ci = 0; ci < in_channels; ++ci) {
            for (std::size_t c = 0; c < nc; ++c) {
                const T* src = xc.plane(static_cast<int>(c)) + ci * h * w;
                T* dst = inp.data() + (ci * nc + c) * plane_in + padding * wpad + padding;
                for (std::size_t y = 0; y < h; ++y) {
                    std::copy(src + y * w, src + (y + 1) * w, dst + y * wpad);
                }
            }
        }

        const std::size_t nblk = (out_channels + CO_BLK - 1) / CO_BLK;
        const std::size_t kk = kh * kw;
        std::vector<T> wpk(in_channels * kk * CO_BLK);
        std::vector<T> outp(CO_BLK * nc * plane_out);
        qtensor_t<T, NT> out({out_channels, ho, wo});
        for (std::size_t b = 0; b < nblk; ++b) {
            std::fill(wpk.begin(), wpk.end(), T(0));
            for (std::size_t a = 0; a < CO_BLK; ++a) {
                const std::size_t co = b * CO_BLK + a;
                if (co >= out_channels) {
                    break;
                }
                const double* wsrc = weight.data() + co * in_channels * kk;
                for (std::size_t i = 0; i < in_channels * kk; ++i) {
                    wpk[i * CO_BLK + a] = static_cast<T>(wsrc[i]);
                }
            }
            conv_detail::block_soa<T>(
                inp.data(), in_channels, hp, wpad, wpk.data(), kh, kw,
                outp.data(), ho, wopad, nc
            );
            for (std::size_t a = 0; a < CO_BLK; ++a) {
                const std::size_t co = b * CO_BLK + a;
                if (co >= out_channels) {
                    break;
                }
                const T bval = bias.empty() ? T(0) : static_cast<T>(bias[co]);
                for (std::size_t c = 0; c < nc; ++c) {
                    const T* src = outp.data() + (a * nc + c) * plane_out;
                    T* dst = out.plane(static_cast<int>(c)) + co * ho * wo;
                    for (std::size_t oy = 0; oy < ho; ++oy) {
                        const T* srow = src + oy * wopad;
                        T* drow = dst + oy * wo;
                        if (c == 0 && bval != T(0)) {
                            for (std::size_t ox = 0; ox < wo; ++ox) {
                                drow[ox] = srow[ox] + bval;
                            }
                        } else {
                            std::copy(srow, srow + wo, drow);
                        }
                    }
                }
            }
        }
        return out;
    }

    template <typename T, int NT>
    qtensor_t<T, NT> forward_generic(
        const qtensor_t<T, NT>& xc, std::size_t h, std::size_t w,
        std::size_t ekh, std::size_t ekw
    ) const {
        constexpr std::size_t nc = qtensor_t<T, NT>::ncomp;
        const std::size_t ho = (h + 2 * padding - ekh) / stride + 1;
        const std::size_t wo = (w + 2 * padding - ekw) / stride + 1;
        qtensor_t<T, NT> out({out_channels, ho, wo});
        const long pad = static_cast<long>(padding);
        for (std::size_t c = 0; c < nc; ++c) {
            const T* src = xc.plane(static_cast<int>(c));
            T* dst = out.plane(static_cast<int>(c));
            for (std::size_t co = 0; co < out_channels; ++co) {
                T* plane = dst + co * ho * wo;
                if (c == 0 && !bias.empty()) {
                    std::fill(plane, plane + ho * wo, static_cast<T>(bias[co]));
                }
                for (std::size_t ci = 0; ci < in_channels; ++ci) {
                    const T* in = src + ci * h * w;
                    const double* wk = weight.data()
                        + (co * in_channels + ci) * kh * kw;
                    for (std::size_t ky = 0; ky < kh; ++ky) {
                        for (std::size_t kx = 0; kx < kw; ++kx) {
                            const T a = static_cast<T>(wk[ky * kw + kx]);
                            if (a == T(0)) {
                                continue;
                            }
                            const long dy = static_cast<long>(ky * dilation) - pad;
                            const long dx = static_cast<long>(kx * dilation) - pad;
                            for (std::size_t oy = 0; oy < ho; ++oy) {
                                const long iy = static_cast<long>(oy * stride) + dy;
                                if (iy < 0 || iy >= static_cast<long>(h)) {
                                    continue;
                                }
                                const T* row = in + iy * w;
                                T* orow = plane + oy * wo;
                                for (std::size_t ox = 0; ox < wo; ++ox) {
                                    const long ix =
                                        static_cast<long>(ox * stride) + dx;
                                    if (ix < 0 || ix >= static_cast<long>(w)) {
                                        continue;
                                    }
                                    orow[ox] += a * row[ix];
                                }
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

    template <typename T, int NT>
    qtensor_t<T, NT> forward(const qtensor_t<T, NT>& x) const {
        if (x.ndim() == 0 || x.shape().back() != in_features) {
            throw std::invalid_argument(
                "Linear: last dimension must be in_features"
            );
        }
        constexpr std::size_t nc = qtensor_t<T, NT>::ncomp;
        const qtensor_t<T, NT> xc = contiguous(x);
        std::vector<std::size_t> shape = xc.shape();
        const std::size_t rows = xc.size() / in_features;
        shape.back() = out_features;
        qtensor_t<T, NT> out(shape);
        for (std::size_t c = 0; c < nc; ++c) {
            const T* src = xc.plane(static_cast<int>(c));
            T* dst = out.plane(static_cast<int>(c));
            for (std::size_t r = 0; r < rows; ++r) {
                const T* in = src + r * in_features;
                T* o = dst + r * out_features;
                for (std::size_t j = 0; j < out_features; ++j) {
                    T acc = (c == 0 && !bias.empty()) ? static_cast<T>(bias[j]) : T(0);
                    const double* wj = weight.data() + j * in_features;
                    for (std::size_t i = 0; i < in_features; ++i) {
                        acc += static_cast<T>(wj[i]) * in[i];
                    }
                    o[j] = acc;
                }
            }
        }
        return out;
    }
};

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_CONV_H
