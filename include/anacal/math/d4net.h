#ifndef ANACAL_MATH_D4NET_H
#define ANACAL_MATH_D4NET_H

#include "../stdafx.h"
#include "qnumber.h"
#include "tensor.h"
#include "conv.h"
#include "activation.h"

// Fused runner of the D4-equivariant shape network (wl_D4T_CNN's
// Forward8_fixW_CNN) on one qnumber stamp: brightness normalisation, the
// 8-element D4 orbit, residual blocks of 3x3 convolution + per-pixel
// channel layer norm + GELU, the inverse orbit with the sign-weighted
// mean, Gaussian-weighted pooling and the two bias-free tanh MLP heads.
// The layers' math is that of conv.h / norm.h / activation.h, but the
// whole forward runs in C++ on persistent structure-of-arrays buffers:
// the 3x3 convolutions use Winograd F(4x4, 3x3) with transforms
// vectorised across tiles (d4net_kernels.h) and the norm / GELU /
// residual of a block are one fused pass.  Returns (e1, e2) with their
// tangents.

namespace anacal {
namespace math {

// no-trapping-math: without it GCC refuses to vectorise loops with
// floor / float-int conversions (the fast exp of the GELU); results do
// not change, the compiler is only told that the operations do not trap.
namespace d4net_detail {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("no-trapping-math")
#endif
namespace generic {
#include "d4net_kernels.h"
} // namespace generic
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define ANACAL_D4NET_HAVE_AVX2 1
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#pragma GCC optimize("no-trapping-math")
namespace avx2 {
#include "d4net_kernels.h"
} // namespace avx2
#pragma GCC pop_options
#else
#define ANACAL_D4NET_HAVE_AVX2 0
#endif
} // namespace d4net_detail

template <typename T>
class D4ShapeNet {
public:
    // One residual block: conv 3x3 (padding 1) + channel norm + GELU,
    // plus res * (skip(x) or x); skip is the 1x1 projection [cout][cin]
    // used when cin != cout (empty otherwise).
    struct Block {
        std::size_t cin = 0;
        std::size_t cout = 0;
        std::vector<T> w;     // [cout][cin][3][3]
        std::vector<T> b;     // [cout]
        std::vector<T> nw;    // [cout] norm weight
        std::vector<T> nb;    // [cout] norm bias
        std::vector<T> skip;  // [cout][cin] or empty
        std::vector<T> upk;   // Winograd weights [36][nblk][cin][CO_BLK]
    };

    D4ShapeNet(
        std::vector<Block> blocks,
        std::vector<std::vector<T>> head_e1,  // 4 bias-free linear layers
        std::vector<std::vector<T>> head_e2,
        std::vector<std::size_t> head_dims,   // [in, h1, h2, h3, out=1]
        std::vector<T> wnorm,                 // [n * n] brightness weights
        std::vector<T> wpool,                 // [n * n] pooling weights
        std::size_t n,
        T res = T(0.1),
        T eps = T(1e-5)
    )
        : blocks_(std::move(blocks)), head_e1_(std::move(head_e1)),
          head_e2_(std::move(head_e2)), head_dims_(std::move(head_dims)),
          wnorm_(std::move(wnorm)), wpool_(std::move(wpool)), n_(n),
          res_(res), eps_(eps),
          use_avx2_(ANACAL_D4NET_HAVE_AVX2 && conv_detail::cpu_has_avx2_fma()) {
        if (blocks_.empty()) {
            throw std::invalid_argument("D4ShapeNet: no blocks");
        }
        if (wnorm_.size() != n_ * n_ || wpool_.size() != n_ * n_) {
            throw std::invalid_argument("D4ShapeNet: weight maps must be n x n");
        }
        if (blocks_.front().cin != 1) {
            throw std::invalid_argument("D4ShapeNet: the first block takes one channel");
        }
        cmax_ = 0;
        for (std::size_t i = 0; i < blocks_.size(); ++i) {
            Block& blk = blocks_[i];
            if (blk.w.size() != blk.cout * blk.cin * 9 || blk.b.size() != blk.cout
                || blk.nw.size() != blk.cout || blk.nb.size() != blk.cout) {
                throw std::invalid_argument("D4ShapeNet: block weight sizes");
            }
            if (blk.cin != blk.cout && blk.skip.size() != blk.cout * blk.cin) {
                throw std::invalid_argument("D4ShapeNet: skip projection size");
            }
            if (i > 0 && blk.cin != blocks_[i - 1].cout) {
                throw std::invalid_argument("D4ShapeNet: channel mismatch");
            }
            cmax_ = std::max(cmax_, std::max(blk.cin, blk.cout));
            pack_winograd(blk);
        }
        wsum_ = T(0);
        for (T a : wpool_) wsum_ += a;
        if (head_e1_.size() != 4 || head_e2_.size() != 4 || head_dims_.size() != 5
            || head_dims_[0] != blocks_.back().cout || head_dims_[4] != 1) {
            throw std::invalid_argument("D4ShapeNet: heads must be 4 linear layers C -> h -> h -> h -> 1");
        }
        for (int j = 0; j < 4; ++j) {
            if (head_e1_[j].size() != head_dims_[j + 1] * head_dims_[j]
                || head_e2_[j].size() != head_dims_[j + 1] * head_dims_[j]) {
                throw std::invalid_argument("D4ShapeNet: head weight sizes");
            }
        }
    }

    bool uses_avx2() const { return use_avx2_; }

    // Stage timings (seconds, accumulated over measure() calls) when
    // profiling is on: orbit, conv_direct, wino_prepare, wino_gemm,
    // wino_output, skip, epilogue, accumulate, heads.
    void set_profile(bool on) { profile_ = on; prof_.fill(0.0); }
    std::array<double, 9> profile() const { return prof_; }

    // (e1, e2) x (v, g1, g2[, x1, x2]) of one [1, n, n] stamp
    template <int NT>
    std::array<T, 2 * (NT + 1)> measure(const qtensor_t<T, NT>& stamp) {
        constexpr std::size_t nc = NT + 1;
        if (stamp.ndim() != 3 || stamp.shape()[0] != 1 || stamp.shape()[1] != n_
            || stamp.shape()[2] != n_) {
            throw std::invalid_argument("D4ShapeNet: stamp must be [1, n, n]");
        }
        const qtensor_t<T, NT> sc = contiguous(stamp);
        const std::size_t npix = n_ * n_;
        ensure_buffers(nc);

        // brightness normalisation: x / (1 + softplus(100 sum w x - 1))
        T m[nc];
        for (std::size_t c = 0; c < nc; ++c) {
            const T* v = sc.plane(static_cast<int>(c));
            T s = T(0);
            for (std::size_t p = 0; p < npix; ++p) s += wnorm_[p] * v[p];
            m[c] = s * T(100);
        }
        T f, df;
        act::softplus(m[0] - T(1), 10.0, 20.0, f, df);
        const T nv = T(1) + f;
        T inv[nc];
        inv[0] = T(1) / nv;
        for (std::size_t c = 1; c < nc; ++c) {
            inv[c] = -(df * m[c]) / (nv * nv);
        }
        // x0 [nc][1][npix]
        {
            const T* v = sc.plane(0);
            for (std::size_t p = 0; p < npix; ++p) x0_[p] = v[p] * inv[0];
            for (std::size_t c = 1; c < nc; ++c) {
                const T* t = sc.plane(static_cast<int>(c));
                T* o = x0_.data() + c * npix;
                for (std::size_t p = 0; p < npix; ++p) o[p] = t[p] * inv[0] + v[p] * inv[c];
            }
        }

        const T s1[8] = {1, -1, 1, -1, 1, -1, 1, -1};
        const T s2[8] = {1, -1, 1, -1, -1, 1, -1, 1};
        const std::size_t C = blocks_.back().cout;
        std::fill(acc1_.begin(), acc1_.begin() + nc * C * npix, T(0));
        std::fill(acc2_.begin(), acc2_.begin() + nc * C * npix, T(0));

        for (int o = 0; o < 8; ++o) {
            // the orbit element: rot90^k, then (o >= 4) a row flip
            auto t0 = tick();
            orbit_forward(x0_.data(), nc, 1, o, X_.data());
            tock(0, t0);
            T* X = X_.data();
            T* Y = Y_.data();
            T* R = R_.data();
            T* OUT = O_.data();
            for (const Block& blk : blocks_) {
                conv3x3(X, nc, blk, Y);
                const T* resid = X;
                if (!blk.skip.empty()) {
                    t0 = tick();
                    skip1x1(X, nc, blk, R);
                    tock(5, t0);
                    resid = R;
                }
                t0 = tick();
                epilogue(Y, resid, OUT, nc, blk);
                tock(6, t0);
                std::swap(X, OUT);
            }
            // X holds the block output: inverse orbit, sign-weighted mean
            t0 = tick();
            if (use_avx2_) {
#if ANACAL_D4NET_HAVE_AVX2
                d4net_detail::avx2::inverse_orbit_accumulate<T>(
                    X, nc * C, n_, o, s1[o] / T(8), s2[o] / T(8), Y, acc1_.data(), acc2_.data()
                );
#endif
            } else {
                d4net_detail::generic::inverse_orbit_accumulate<T>(
                    X, nc * C, n_, o, s1[o] / T(8), s2[o] / T(8), Y, acc1_.data(), acc2_.data()
                );
            }
            tock(7, t0);
        }

        std::array<T, 2 * nc> result{};
        auto t0 = tick();
        head(acc1_.data(), nc, C, head_e1_, result.data());
        head(acc2_.data(), nc, C, head_e2_, result.data() + nc);
        tock(8, t0);
        return result;
    }

private:
    std::vector<Block> blocks_;
    std::vector<std::vector<T>> head_e1_, head_e2_;
    std::vector<std::size_t> head_dims_;
    std::vector<T> wnorm_, wpool_;
    std::size_t n_;
    T res_, eps_, wsum_ = T(1);
    bool use_avx2_;
    std::size_t cmax_ = 0;
    // the GEMM column tile: two 32-byte vectors
    static constexpr std::size_t wino_cols_ = 2 * (32 / sizeof(T));
    bool profile_ = false;
    std::array<double, 9> prof_{};
    using clock = std::chrono::steady_clock;
    clock::time_point tick() const { return profile_ ? clock::now() : clock::time_point(); }
    void tock(int stage, clock::time_point t0) {
        if (profile_) {
            prof_[stage] += std::chrono::duration<double>(clock::now() - t0).count();
        }
    }
    // work buffers
    std::vector<T> x0_, X_, Y_, R_, O_, acc1_, acc2_;
    std::vector<T> pad_, V_, M_, wpk_, t1_, ph_, z_, yy_, stat_;

    void ensure_buffers(std::size_t nc) {
        const std::size_t npix = n_ * n_;
        x0_.resize(nc * npix);
        X_.resize(nc * cmax_ * npix);
        Y_.resize(nc * cmax_ * npix);
        R_.resize(nc * cmax_ * npix);
        O_.resize(nc * cmax_ * npix);
        acc1_.resize(nc * cmax_ * npix);
        acc2_.resize(nc * cmax_ * npix);
        const std::size_t th = (n_ + d4net_detail::generic::WO - 1) / d4net_detail::generic::WO;
        const std::size_t tw = th;
        const std::size_t hp = th * 4 + 2, wp = tw * 4 + 2;
        const std::size_t ntiles = th * tw;
        const std::size_t ncol = nc * ntiles;
        const std::size_t ncolpad = (ncol + wino_cols_ - 1) / wino_cols_ * wino_cols_;
        const std::size_t cpad = (cmax_ + conv_detail::CO_BLK - 1) / conv_detail::CO_BLK * conv_detail::CO_BLK;
        pad_.resize(cmax_ * nc * hp * wp);
        V_.resize(36 * cmax_ * ncolpad);
        M_.resize(36 * cpad * ncolpad);
        t1_.resize(th * 6 * (wp + 2));
        ph_.resize(6 * ntiles);
        z_.resize(6 * 4 * ncol);
        yy_.resize(4 * 4 * ncol);
        stat_.resize((2 * nc + 1) * npix);
        // direct conv of the first block: padded input and packed weights
        const std::size_t wopad = (n_ + conv_detail::PX - 1) / conv_detail::PX * conv_detail::PX;
        pad_.resize(std::max(pad_.size(), cmax_ * nc * (n_ + 2) * (wopad + 2)));
        M_.resize(std::max(M_.size(), conv_detail::CO_BLK * nc * n_ * wopad));
        wpk_.resize(cmax_ * 9 * conv_detail::CO_BLK);
    }

    // Winograd-transformed, block-packed weights of a block
    void pack_winograd(Block& blk) {
        using namespace d4net_detail::generic;
        using M = wino_mats<T>;
        const std::size_t nblk = (blk.cout + conv_detail::CO_BLK - 1) / conv_detail::CO_BLK;
        blk.upk.assign(NK * nblk * blk.cin * conv_detail::CO_BLK, T(0));
        T tmp[6][3], U[6][6];
        for (std::size_t co = 0; co < blk.cout; ++co) {
            const std::size_t b = co / conv_detail::CO_BLK, a = co % conv_detail::CO_BLK;
            for (std::size_t ci = 0; ci < blk.cin; ++ci) {
                const T* g = blk.w.data() + (co * blk.cin + ci) * 9;
                for (std::size_t i = 0; i < 6; ++i) {
                    for (std::size_t j = 0; j < 3; ++j) {
                        tmp[i][j] = M::G[i][0] * g[j] + M::G[i][1] * g[3 + j] + M::G[i][2] * g[6 + j];
                    }
                }
                for (std::size_t i = 0; i < 6; ++i) {
                    for (std::size_t j = 0; j < 6; ++j) {
                        U[i][j] = tmp[i][0] * M::G[j][0] + tmp[i][1] * M::G[j][1] + tmp[i][2] * M::G[j][2];
                    }
                }
                for (std::size_t k = 0; k < NK; ++k) {
                    blk.upk[((k * nblk + b) * blk.cin + ci) * conv_detail::CO_BLK + a] = U[k / 6][k % 6];
                }
            }
        }
    }

    // out[c][co][p] of the orbit element o applied to in[c][co][p]
    void orbit_forward(const T* in, std::size_t nc, std::size_t C, int o, T* out) const {
        const int k = o % 4;
        const bool flip = o >= 4;
        const std::size_t n = n_;
        for (std::size_t pl = 0; pl < nc * C; ++pl) {
            const T* src = in + pl * n * n;
            T* dst = out + pl * n * n;
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t ii = flip ? n - 1 - i : i;  // flip rows AFTER the rotation
                for (std::size_t j = 0; j < n; ++j) {
                    std::size_t y = ii, x = j;
                    if (k == 1) { y = j; x = n - 1 - ii; }
                    else if (k == 2) { y = n - 1 - ii; x = n - 1 - j; }
                    else if (k == 3) { y = n - 1 - j; x = ii; }
                    dst[i * n + j] = src[y * n + x];
                }
            }
        }
    }

    // acc += s * (inverse orbit element of F), F [nc][C][n][n]
    void accumulate_inverse(const T* F, std::size_t nc, std::size_t C, int o, T sa, T sb) {
        const int k = o % 4;
        const bool flip = o >= 4;
        const std::size_t n = n_;
        for (std::size_t pl = 0; pl < nc * C; ++pl) {
            const T* src = F + pl * n * n;
            T* a1 = acc1_.data() + pl * n * n;
            T* a2 = acc2_.data() + pl * n * n;
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t ii = flip ? n - 1 - i : i;
                for (std::size_t j = 0; j < n; ++j) {
                    // the forward map sent canonical (y, x) to rotated (ii, j)
                    std::size_t y = ii, x = j;
                    if (k == 1) { y = j; x = n - 1 - ii; }
                    else if (k == 2) { y = n - 1 - ii; x = n - 1 - j; }
                    else if (k == 3) { y = n - 1 - j; x = ii; }
                    const T val = src[i * n + j];
                    a1[y * n + x] += sa * val;
                    a2[y * n + x] += sb * val;
                }
            }
        }
    }

    // 1x1 projection R[c][co][p] = sum_ci skip[co][ci] X[c][ci][p]
    void skip1x1(const T* X, std::size_t nc, const Block& blk, T* R) const {
        const std::size_t npix = n_ * n_;
        for (std::size_t c = 0; c < nc; ++c) {
            for (std::size_t co = 0; co < blk.cout; ++co) {
                T* o = R + (c * blk.cout + co) * npix;
                for (std::size_t p = 0; p < npix; ++p) o[p] = T(0);
                for (std::size_t ci = 0; ci < blk.cin; ++ci) {
                    const T w = blk.skip[co * blk.cin + ci];
                    const T* x = X + (c * blk.cin + ci) * npix;
                    for (std::size_t p = 0; p < npix; ++p) o[p] += w * x[p];
                }
            }
        }
    }

    void conv3x3(const T* X, std::size_t nc, const Block& blk, T* Y) {
        if (blk.cin >= 4) {
            conv3x3_winograd(X, nc, blk, Y);
        } else {
            auto t0 = tick();
            conv3x3_direct(X, nc, blk, Y);
            tock(1, t0);
        }
    }

    // X [nc][cin][n][n] -> Y [nc][cout][n][n], bias on the value planes
    void conv3x3_winograd(const T* X, std::size_t nc, const Block& blk, T* Y) {
        using namespace d4net_detail::generic;  // the constants
        const std::size_t n = n_;
        const std::size_t th = (n + WO - 1) / WO, tw = th;
        const std::size_t hp = th * WO + 2, wp = tw * WO + 2;
        const std::size_t ntiles = th * tw;
        const std::size_t nplane = blk.cin * nc;
        const std::size_t ncol = nc * ntiles;
        const std::size_t ncolpad = (ncol + wino_cols_ - 1) / wino_cols_ * wino_cols_;
        const std::size_t nblk = (blk.cout + conv_detail::CO_BLK - 1) / conv_detail::CO_BLK;
        const std::size_t cpad = nblk * conv_detail::CO_BLK;
        // padded planes, plane index pl = ci * nc + c (channel-major so a
        // GEMM row is one input channel with the components as columns)
        auto t0 = tick();
        std::fill(pad_.begin(), pad_.begin() + nplane * hp * wp, T(0));
        for (std::size_t ci = 0; ci < blk.cin; ++ci) {
            for (std::size_t c = 0; c < nc; ++c) {
                const T* src = X + (c * blk.cin + ci) * n * n;
                T* dst = pad_.data() + (ci * nc + c) * hp * wp + wp + 1;
                for (std::size_t y = 0; y < n; ++y) {
                    std::copy(src + y * n, src + (y + 1) * n, dst + y * wp);
                }
            }
        }
        // V[k][ci][col], col = c * ntiles + tile: the input transform fills
        // V[k][pl][tile] with pl = ci * nc + c and plane stride ntiles, i.e.
        // exactly [k][ci][c * ntiles + tile] when ncolpad == nc * ntiles;
        // with padding, transform per channel into the right row.
        if (ncolpad != ncol) {
            // the register tile reads past the last column: keep those zero
            for (std::size_t k = 0; k < NK; ++k) {
                for (std::size_t ci = 0; ci < blk.cin; ++ci) {
                    T* row = V_.data() + (k * blk.cin + ci) * ncolpad;
                    std::fill(row + ncol, row + ncolpad, T(0));
                }
            }
        }
        for (std::size_t ci = 0; ci < blk.cin; ++ci) {
            const T* P = pad_.data() + ci * nc * hp * wp;
            // the nc planes of channel ci occupy columns [0, nc * ntiles) of row ci
            T* Vci = V_.data() + ci * ncolpad;
            if (use_avx2_) {
#if ANACAL_D4NET_HAVE_AVX2
                d4net_detail::avx2::wino_input<T>(P, nc, hp, wp, th, tw, Vci, blk.cin * ncolpad, t1_.data(), ph_.data());
#endif
            } else {
                d4net_detail::generic::wino_input<T>(P, nc, hp, wp, th, tw, Vci, blk.cin * ncolpad, t1_.data(), ph_.data());
            }
        }
        tock(2, t0);
        // M[k][co][col] = sum_ci U[k][co][ci] V[k][ci][col]
        t0 = tick();
        for (std::size_t k = 0; k < NK; ++k) {
            const T* Vk = V_.data() + k * blk.cin * ncolpad;
            for (std::size_t b = 0; b < nblk; ++b) {
                const T* upk = blk.upk.data() + (k * nblk + b) * blk.cin * conv_detail::CO_BLK;
                T* Mk = M_.data() + (k * cpad + b * conv_detail::CO_BLK) * ncolpad;
                if (use_avx2_) {
#if ANACAL_D4NET_HAVE_AVX2
                    d4net_detail::avx2::gemm_block<T>(Vk, blk.cin, ncolpad, upk, Mk);
                    continue;
#endif
                }
                d4net_detail::generic::gemm_block<T>(Vk, blk.cin, ncolpad, upk, Mk);
            }
        }
        tock(3, t0);
        t0 = tick();
        if (use_avx2_) {
#if ANACAL_D4NET_HAVE_AVX2
            d4net_detail::avx2::wino_output<T>(
                M_.data(), blk.cout, cpad, ncolpad, nc, nc, th, tw, n, blk.b.data(),
                Y, z_.data(), yy_.data()
            );
#endif
        } else {
            d4net_detail::generic::wino_output<T>(
                M_.data(), blk.cout, cpad, ncolpad, nc, nc, th, tw, n, blk.b.data(),
                Y, z_.data(), yy_.data()
            );
        }
        tock(4, t0);
    }

    // direct 3x3 / padding 1 through block_soa (the one-channel first block)
    void conv3x3_direct(const T* X, std::size_t nc, const Block& blk, T* Y) {
        using conv_detail::CO_BLK;
        using conv_detail::PX;
        const std::size_t n = n_;
        const std::size_t hp = n + 2;
        const std::size_t wopad = (n + PX - 1) / PX * PX;
        const std::size_t wpad = wopad + 2;
        const std::size_t plane_in = hp * wpad;
        const std::size_t plane_out = n * wopad;
        std::fill(pad_.begin(), pad_.begin() + blk.cin * nc * plane_in, T(0));
        for (std::size_t ci = 0; ci < blk.cin; ++ci) {
            for (std::size_t c = 0; c < nc; ++c) {
                const T* src = X + (c * blk.cin + ci) * n * n;
                T* dst = pad_.data() + (ci * nc + c) * plane_in + wpad + 1;
                for (std::size_t y = 0; y < n; ++y) {
                    std::copy(src + y * n, src + (y + 1) * n, dst + y * wpad);
                }
            }
        }
        const std::size_t nblk = (blk.cout + CO_BLK - 1) / CO_BLK;
        for (std::size_t b = 0; b < nblk; ++b) {
            std::fill(wpk_.begin(), wpk_.begin() + blk.cin * 9 * CO_BLK, T(0));
            for (std::size_t a = 0; a < CO_BLK; ++a) {
                const std::size_t co = b * CO_BLK + a;
                if (co >= blk.cout) break;
                const T* w = blk.w.data() + co * blk.cin * 9;
                for (std::size_t i = 0; i < blk.cin * 9; ++i) wpk_[i * CO_BLK + a] = w[i];
            }
            conv_detail::block_soa<T>(
                pad_.data(), blk.cin, hp, wpad, wpk_.data(), 3, 3, M_.data(), n, wopad, nc
            );
            for (std::size_t a = 0; a < CO_BLK; ++a) {
                const std::size_t co = b * CO_BLK + a;
                if (co >= blk.cout) break;
                for (std::size_t c = 0; c < nc; ++c) {
                    const T* src = M_.data() + (a * nc + c) * plane_out;
                    T* dst = Y + (c * blk.cout + co) * n * n;
                    const T add = (c == 0) ? blk.b[co] : T(0);
                    for (std::size_t y = 0; y < n; ++y) {
                        for (std::size_t x = 0; x < n; ++x) dst[y * n + x] = src[y * wopad + x] + add;
                    }
                }
            }
        }
    }

    void epilogue(const T* Y, const T* R, T* OUT, std::size_t nc, const Block& blk) {
        if (use_avx2_) {
#if ANACAL_D4NET_HAVE_AVX2
            d4net_detail::avx2::block_epilogue<T>(
                Y, R, OUT, nc, blk.cout, n_ * n_, blk.nw.data(), blk.nb.data(), eps_, res_,
                stat_.data()
            );
            return;
#endif
        }
        d4net_detail::generic::block_epilogue<T>(
            Y, R, OUT, nc, blk.cout, n_ * n_, blk.nw.data(), blk.nb.data(), eps_, res_,
            stat_.data()
        );
    }

    // pooled features -> bias-free tanh MLP -> one qnumber (nc components)
    void head(const T* acc, std::size_t nc, std::size_t C, const std::vector<std::vector<T>>& lin, T* out) const {
        const std::size_t npix = n_ * n_;
        std::vector<T> v(nc * C), w;
        for (std::size_t c = 0; c < nc; ++c) {
            for (std::size_t ch = 0; ch < C; ++ch) {
                const T* a = acc + (c * C + ch) * npix;
                T s = T(0);
                for (std::size_t p = 0; p < npix; ++p) s += a[p] * wpool_[p];
                v[c * C + ch] = s / wsum_;
            }
        }
        std::size_t din = C;
        for (int j = 0; j < 4; ++j) {
            const std::size_t dout = head_dims_[j + 1];
            w.assign(nc * dout, T(0));
            for (std::size_t c = 0; c < nc; ++c) {
                for (std::size_t o = 0; o < dout; ++o) {
                    T s = T(0);
                    const T* wj = lin[j].data() + o * din;
                    for (std::size_t i = 0; i < din; ++i) s += wj[i] * v[c * din + i];
                    w[c * dout + o] = s;
                }
            }
            if (j < 3) {
                for (std::size_t o = 0; o < dout; ++o) {
                    const T t = std::tanh(w[o]);
                    const T dt = T(1) - t * t;
                    w[o] = t;
                    for (std::size_t c = 1; c < nc; ++c) w[c * dout + o] *= dt;
                }
            }
            v.swap(w);
            din = dout;
        }
        for (std::size_t c = 0; c < nc; ++c) out[c] = v[c];
    }
};

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_D4NET_H
