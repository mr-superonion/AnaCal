#ifndef ANACAL_MATH_TRANSFORM_H
#define ANACAL_MATH_TRANSFORM_H

#include "../stdafx.h"
#include "qnumber.h"
#include "tensor.h"

// Pixel rearrangements of [C, H, W] qnumber tensors (D4 orbit, crop,
// pad).  They move whole qnumbers: the tangent slots are derivatives of
// each pixel with respect to the global shear and position, so they
// travel with the pixel and are not rotated themselves.  Each
// rearrangement is applied plane by plane.

namespace anacal {
namespace math {

namespace detail {
template <typename T, int NT>
inline void check_spatial(const qtensor_t<T, NT>& x, const char* name) {
    if (x.ndim() != 3) {
        throw std::invalid_argument(
            std::string(name) + ": input must be [C, H, W]"
        );
    }
}
} // namespace detail

// torch.rot90(x, k, dims=(-2, -1)): counter-clockwise by k quarter turns
template <typename T, int NT>
inline qtensor_t<T, NT> rot90(const qtensor_t<T, NT>& x, int k = 1) {
    detail::check_spatial(x, "rot90");
    k = ((k % 4) + 4) % 4;
    const qtensor_t<T, NT> xc = contiguous(x);
    const std::size_t c = xc.shape()[0];
    const std::size_t h = xc.shape()[1], w = xc.shape()[2];
    const std::size_t ho = (k % 2) ? w : h;
    const std::size_t wo = (k % 2) ? h : w;
    qtensor_t<T, NT> out({c, ho, wo});
    for (int comp = 0; comp < qtensor_t<T, NT>::ncomp; ++comp) {
        const T* src = xc.plane(comp);
        T* dst = out.plane(comp);
        for (std::size_t ic = 0; ic < c; ++ic) {
            const T* in = src + ic * h * w;
            T* o = dst + ic * ho * wo;
            for (std::size_t i = 0; i < ho; ++i) {
                for (std::size_t j = 0; j < wo; ++j) {
                    std::size_t y = i, xx = j;
                    if (k == 1) {
                        y = j;
                        xx = w - 1 - i;
                    } else if (k == 2) {
                        y = h - 1 - i;
                        xx = w - 1 - j;
                    } else if (k == 3) {
                        y = h - 1 - j;
                        xx = i;
                    }
                    o[i * wo + j] = in[y * w + xx];
                }
            }
        }
    }
    return out;
}

// torch.flip(x, dims=[dim]) for dim = 1 (rows, the D4 CNN's reflection
// torch.flip(x, [-2])) or dim = 2 (columns)
template <typename T, int NT>
inline qtensor_t<T, NT> flip(const qtensor_t<T, NT>& x, std::size_t dim) {
    detail::check_spatial(x, "flip");
    if (dim != 1 && dim != 2) {
        throw std::invalid_argument("flip: dim must be 1 or 2");
    }
    const qtensor_t<T, NT> xc = contiguous(x);
    const std::size_t c = xc.shape()[0];
    const std::size_t h = xc.shape()[1], w = xc.shape()[2];
    qtensor_t<T, NT> out({c, h, w});
    for (int comp = 0; comp < qtensor_t<T, NT>::ncomp; ++comp) {
        const T* src = xc.plane(comp);
        T* dst = out.plane(comp);
        for (std::size_t ic = 0; ic < c; ++ic) {
            for (std::size_t i = 0; i < h; ++i) {
                const std::size_t y = (dim == 1) ? h - 1 - i : i;
                const T* in = src + (ic * h + y) * w;
                T* o = dst + (ic * h + i) * w;
                if (dim == 1) {
                    std::copy(in, in + w, o);
                } else {
                    for (std::size_t j = 0; j < w; ++j) {
                        o[j] = in[w - 1 - j];
                    }
                }
            }
        }
    }
    return out;
}

// x[:, y0:y0 + h, x0:x0 + w] as a contiguous tensor
template <typename T, int NT>
inline qtensor_t<T, NT> crop2d(
    const qtensor_t<T, NT>& x, std::size_t y0, std::size_t x0,
    std::size_t h, std::size_t w
) {
    detail::check_spatial(x, "crop2d");
    if (y0 + h > x.shape()[1] || x0 + w > x.shape()[2]) {
        throw std::out_of_range("crop2d: window outside the input");
    }
    return contiguous(x.slice(1, y0, y0 + h).slice(2, x0, x0 + w));
}

// F.pad(x, (p, p, p, p), mode) for mode "constant" (zeros) or
// "replicate"
template <typename T, int NT>
inline qtensor_t<T, NT> pad2d(
    const qtensor_t<T, NT>& x, std::size_t p, const std::string& mode = "constant"
) {
    detail::check_spatial(x, "pad2d");
    const bool replicate = (mode == "replicate");
    if (!replicate && mode != "constant") {
        throw std::invalid_argument("pad2d: mode must be constant or replicate");
    }
    const qtensor_t<T, NT> xc = contiguous(x);
    const std::size_t c = xc.shape()[0];
    const std::size_t h = xc.shape()[1], w = xc.shape()[2];
    if (replicate && (h == 0 || w == 0)) {
        throw std::invalid_argument("pad2d: cannot replicate an empty input");
    }
    const std::size_t ho = h + 2 * p, wo = w + 2 * p;
    qtensor_t<T, NT> out({c, ho, wo});
    const long lp = static_cast<long>(p);
    for (int comp = 0; comp < qtensor_t<T, NT>::ncomp; ++comp) {
        const T* src = xc.plane(comp);
        T* dst = out.plane(comp);
        for (std::size_t ic = 0; ic < c; ++ic) {
            for (std::size_t i = 0; i < ho; ++i) {
                long y = static_cast<long>(i) - lp;
                for (std::size_t j = 0; j < wo; ++j) {
                    long xx = static_cast<long>(j) - lp;
                    const bool inside = y >= 0 && y < static_cast<long>(h)
                        && xx >= 0 && xx < static_cast<long>(w);
                    if (!inside && !replicate) {
                        continue;
                    }
                    const long yc = std::clamp(y, 0L, static_cast<long>(h) - 1);
                    const long xcl = std::clamp(xx, 0L, static_cast<long>(w) - 1);
                    dst[(ic * ho + i) * wo + j] = src[(ic * h + yc) * w + xcl];
                }
            }
        }
    }
    return out;
}

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_TRANSFORM_H
