#ifndef ANACAL_MATH_POOLING_H
#define ANACAL_MATH_POOLING_H

#include "../stdafx.h"
#include "qnumber.h"
#include "tensor.h"

namespace anacal {
namespace math {

    namespace detail {
        inline double scalar_of(double t) { return t; }
        inline double scalar_of(const qnumber& t) { return t.v; }
    }

    // Shared clamp shell of the smooth-step family: map x onto
    // t = (x - mu) / (2 sigma) + 1/2, return 0 below the ramp, 1 above
    // it, and the supplied polynomial on it.  Each public variant passes
    // its ORIGINAL polynomial expression, so floating-point behaviour is
    // exactly what the six hand-written copies produced.
    template <typename X, typename M, typename Poly>
    inline auto ssfunc_shell(const X& x, M mu, M sigma, Poly poly)
        -> decltype(poly((x - mu) / (sigma * 2.0) + 0.5)) {
        auto t = (x - mu) / (sigma * 2.0) + 0.5;
        using R = decltype(poly(t));
        if (detail::scalar_of(t) < 0) return R(0.0);
        else if (detail::scalar_of(t) <= 1) return poly(t);
        else return R(1.0);
    }

    // Same shell for the derivatives: zero outside the ramp on BOTH
    // sides, and the chain-rule factor 1 / (2 sigma) inside it.
    template <typename Poly>
    inline double ssfunc_deriv_shell(
        double x, double mu, double sigma, Poly poly
    ) {
        double t = (x - mu) / (sigma * 2.0) + 0.5;
        if (t < 0) return 0.0;
        else if (t <= 1) return poly(t) / (sigma * 2.0);
        else return 0.0;
    }

    // C1 smooth step
    inline qnumber ssfunc1(qnumber x, double mu, double sigma) {
        return ssfunc_shell(x, mu, sigma, [](const qnumber& t) {
            return -2.0 * pow(t, 3.0) + 3.0 * pow(t, 2);
        });
    }

    inline double ssfunc1(double x, double mu, double sigma) {
        return ssfunc_shell(x, mu, sigma, [](double t) {
            return -2.0 * t * t * t + 3 * t * t;
        });
    }

    inline double ssfunc1_deriv(double x, double mu, double sigma) {
        return ssfunc_deriv_shell(x, mu, sigma, [](double t) {
            return -6.0 * t * t + 6 * t;
        });
    }

    // C2 smooth step
    template <typename T>
    inline qnumber ssfunc2(qnumber x, T mu, T sigma) {
        return ssfunc_shell(x, mu, sigma, [](const qnumber& t) {
            return 6 * pow(t, 5) - 15 * pow(t, 4) + 10 * pow(t, 3);
        });
    }

    inline double ssfunc2(double x, double mu, double sigma) {
        return ssfunc_shell(x, mu, sigma, [](double t) {
            double t3 = t * t * t;
            return (6 * t * t - 15 * t + 10) * t3;
        });
    }

    inline double ssfunc2_deriv(double x, double mu, double sigma) {
        return ssfunc_deriv_shell(x, mu, sigma, [](double t) {
            double t2 = t * t;
            return (30 * t2 - 60 * t + 30) * t2;
        });
    }


    // ------------------------------------------------------------------
    // Pooling layers of a network on [C, H, W] qnumber tensors, plane by
    // plane
    // ------------------------------------------------------------------

    namespace detail {
        inline std::size_t pool_extent(
            std::size_t n, std::size_t k, std::size_t stride,
            std::size_t padding
        ) {
            if (k == 0 || stride == 0 || n + 2 * padding < k) {
                throw std::invalid_argument("pooling: invalid kernel");
            }
            return (n + 2 * padding - k) / stride + 1;
        }

        template <typename T, int NT>
        inline void check_chw(const qtensor_t<T, NT>& x, const char* name) {
            if (x.ndim() != 3) {
                throw std::invalid_argument(
                    std::string(name) + ": input must be [C, H, W]"
                );
            }
        }
    }

    // torch.nn.MaxPool2d (stride 0 = kernel size, padding with -inf):
    // the output is the qnumber of the window's largest value, so its
    // tangents are those of the arg-max pixel (the first one on ties)
    template <typename T, int NT>
    inline qtensor_t<T, NT> max_pool2d(
        const qtensor_t<T, NT>& x, std::size_t k, std::size_t stride = 0,
        std::size_t padding = 0
    ) {
        detail::check_chw(x, "max_pool2d");
        if (stride == 0) stride = k;
        if (2 * padding > k) {
            throw std::invalid_argument(
                "max_pool2d: padding must be at most half the kernel"
            );
        }
        const qtensor_t<T, NT> xc = contiguous(x);
        const std::size_t c = xc.shape()[0];
        const std::size_t h = xc.shape()[1], w = xc.shape()[2];
        const std::size_t ho = detail::pool_extent(h, k, stride, padding);
        const std::size_t wo = detail::pool_extent(w, k, stride, padding);
        qtensor_t<T, NT> out({c, ho, wo});
        const T* v = xc.plane(0);
        const long pad = static_cast<long>(padding);
        for (std::size_t ic = 0; ic < c; ++ic) {
            const T* in = v + ic * h * w;
            for (std::size_t oy = 0; oy < ho; ++oy) {
                for (std::size_t ox = 0; ox < wo; ++ox) {
                    long best = -1;
                    for (std::size_t ky = 0; ky < k; ++ky) {
                        const long iy = static_cast<long>(oy * stride + ky) - pad;
                        if (iy < 0 || iy >= static_cast<long>(h)) continue;
                        for (std::size_t kx = 0; kx < k; ++kx) {
                            const long ix =
                                static_cast<long>(ox * stride + kx) - pad;
                            if (ix < 0 || ix >= static_cast<long>(w)) continue;
                            const long idx = iy * static_cast<long>(w) + ix;
                            if (best < 0 || in[idx] > in[best]) best = idx;
                        }
                    }
                    const std::size_t o = (ic * ho + oy) * wo + ox;
                    const std::size_t src = ic * h * w + static_cast<std::size_t>(best);
                    for (int comp = 0; comp < qtensor_t<T, NT>::ncomp; ++comp) {
                        out.plane(comp)[o] = xc.plane(comp)[src];
                    }
                }
            }
        }
        return out;
    }

    // torch.nn.AvgPool2d (stride 0 = kernel size, zero padding counted in
    // the mean when count_include_pad)
    template <typename T, int NT>
    inline qtensor_t<T, NT> avg_pool2d(
        const qtensor_t<T, NT>& x, std::size_t k, std::size_t stride = 0,
        std::size_t padding = 0, bool count_include_pad = true
    ) {
        detail::check_chw(x, "avg_pool2d");
        if (stride == 0) stride = k;
        if (2 * padding > k) {
            throw std::invalid_argument(
                "avg_pool2d: padding must be at most half the kernel"
            );
        }
        const qtensor_t<T, NT> xc = contiguous(x);
        const std::size_t c = xc.shape()[0];
        const std::size_t h = xc.shape()[1], w = xc.shape()[2];
        const std::size_t ho = detail::pool_extent(h, k, stride, padding);
        const std::size_t wo = detail::pool_extent(w, k, stride, padding);
        qtensor_t<T, NT> out({c, ho, wo});
        const long pad = static_cast<long>(padding);
        for (int comp = 0; comp < qtensor_t<T, NT>::ncomp; ++comp) {
            const T* src = xc.plane(comp);
            T* dst = out.plane(comp);
            for (std::size_t ic = 0; ic < c; ++ic) {
                const T* in = src + ic * h * w;
                for (std::size_t oy = 0; oy < ho; ++oy) {
                    for (std::size_t ox = 0; ox < wo; ++ox) {
                        T acc = 0;
                        std::size_t count = 0;
                        for (std::size_t ky = 0; ky < k; ++ky) {
                            const long iy = static_cast<long>(oy * stride + ky) - pad;
                            for (std::size_t kx = 0; kx < k; ++kx) {
                                const long ix =
                                    static_cast<long>(ox * stride + kx) - pad;
                                // PyTorch counts padded cells, not cells past
                                // the padding (impossible with 2 padding <= k)
                                if (iy < 0 || iy >= static_cast<long>(h)
                                    || ix < 0 || ix >= static_cast<long>(w)) {
                                    if (count_include_pad) ++count;
                                    continue;
                                }
                                acc += in[iy * w + ix];
                                ++count;
                            }
                        }
                        dst[(ic * ho + oy) * wo + ox] = acc / static_cast<T>(count);
                    }
                }
            }
        }
        return out;
    }

    // torch.nn.AdaptiveAvgPool2d((oh, ow)): bin i covers
    // [floor(i n / o), ceil((i + 1) n / o))
    template <typename T, int NT>
    inline qtensor_t<T, NT> adaptive_avg_pool2d(
        const qtensor_t<T, NT>& x, std::size_t oh, std::size_t ow
    ) {
        detail::check_chw(x, "adaptive_avg_pool2d");
        if (oh == 0 || ow == 0) {
            throw std::invalid_argument("adaptive_avg_pool2d: zero output");
        }
        const qtensor_t<T, NT> xc = contiguous(x);
        const std::size_t c = xc.shape()[0];
        const std::size_t h = xc.shape()[1], w = xc.shape()[2];
        qtensor_t<T, NT> out({c, oh, ow});
        for (int comp = 0; comp < qtensor_t<T, NT>::ncomp; ++comp) {
            const T* src = xc.plane(comp);
            T* dst = out.plane(comp);
            for (std::size_t ic = 0; ic < c; ++ic) {
                const T* in = src + ic * h * w;
                for (std::size_t oy = 0; oy < oh; ++oy) {
                    const std::size_t y0 = (oy * h) / oh;
                    const std::size_t y1 = ((oy + 1) * h + oh - 1) / oh;
                    for (std::size_t ox = 0; ox < ow; ++ox) {
                        const std::size_t x0 = (ox * w) / ow;
                        const std::size_t x1 = ((ox + 1) * w + ow - 1) / ow;
                        T acc = 0;
                        for (std::size_t iy = y0; iy < y1; ++iy) {
                            for (std::size_t ix = x0; ix < x1; ++ix) {
                                acc += in[iy * w + ix];
                            }
                        }
                        dst[(ic * oh + oy) * ow + ox] =
                            acc / static_cast<T>((y1 - y0) * (x1 - x0));
                    }
                }
            }
        }
        return out;
    }

    // Weighted global average pool: [C, H, W] -> [C],
    // sum_p x[c, p] w[p] / sum_p w[p] (sum_p x[c, p] w[p] without
    // normalize), with a fixed (image-independent) weight map w [H, W];
    // uniform weights give the global average pool
    template <typename T, int NT>
    inline qtensor_t<T, NT> weighted_avg_pool(
        const qtensor_t<T, NT>& x, const std::vector<double>& weight,
        bool normalize = true
    ) {
        detail::check_chw(x, "weighted_avg_pool");
        const qtensor_t<T, NT> xc = contiguous(x);
        const std::size_t c = xc.shape()[0];
        const std::size_t npix = xc.shape()[1] * xc.shape()[2];
        if (weight.size() != npix) {
            throw std::invalid_argument(
                "weighted_avg_pool: weight must have H * W elements"
            );
        }
        double wsum = 1.0;
        if (normalize) {
            wsum = 0.0;
            for (double a : weight) wsum += a;
            if (wsum == 0.0) {
                throw std::invalid_argument("weighted_avg_pool: zero weight");
            }
        }
        std::vector<T> wt(npix);
        for (std::size_t i = 0; i < npix; ++i) {
            wt[i] = static_cast<T>(weight[i]);
        }
        const T inv = static_cast<T>(1.0 / wsum);
        qtensor_t<T, NT> out({c});
        for (int comp = 0; comp < qtensor_t<T, NT>::ncomp; ++comp) {
            const T* src = xc.plane(comp);
            T* dst = out.plane(comp);
            for (std::size_t ic = 0; ic < c; ++ic) {
                const T* in = src + ic * npix;
                T acc = 0;
                for (std::size_t i = 0; i < npix; ++i) {
                    acc += wt[i] * in[i];
                }
                dst[ic] = acc * inv;
            }
        }
        return out;
    }

    // torch.nn.AdaptiveAvgPool2d(1) followed by a flatten: [C, H, W] -> [C]
    template <typename T, int NT>
    inline qtensor_t<T, NT> global_avg_pool(const qtensor_t<T, NT>& x) {
        detail::check_chw(x, "global_avg_pool");
        return weighted_avg_pool(
            x, std::vector<double>(x.shape()[1] * x.shape()[2], 1.0), true
        );
    }

}
}

#endif // ANACAL_MATH_POOLING_H
