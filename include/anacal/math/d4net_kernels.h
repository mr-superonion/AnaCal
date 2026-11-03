// Kernels of the fused D4 shape-network runner (d4net.h).  This header has
// no include guard on purpose: d4net.h includes it twice, once in
// namespace d4net_detail::generic and once in d4net_detail::avx2 under
// `#pragma GCC target("avx2,fma")`, so every loop below -- the Winograd
// transforms, the fused normalisation / GELU / residual epilogue -- is
// compiled for AVX2 as well as for the baseline ISA, and the runner picks
// one set at run time.  Nothing here may call an inline function defined
// outside this header (it could not be inlined into the AVX2 clone), which
// is why the float exp / GELU are repeated here.

// ---------------------------------------------------------------------
// Winograd F(4x4, 3x3): Y = A^T [(G g G^T) o (B^T d B)] A on 4x4 output
// tiles from 6x6 input tiles (stride 4, padding 1).  The transforms run
// over whole rows of tiles with the tile index innermost, so they
// vectorise; the 36 element-wise products over the input channels are
// matrix products done by conv_detail::block_soa.
// ---------------------------------------------------------------------

constexpr std::size_t WT = 6;   // input tile
constexpr std::size_t WO = 4;   // output tile
constexpr std::size_t NK = WT * WT;

// B^T, G, A^T of F(4x4, 3x3)
template <typename T>
struct wino_mats {
    static constexpr T BT[6][6] = {
        {4, 0, -5, 0, 1, 0},
        {0, -4, -4, 1, 1, 0},
        {0, 4, -4, -1, 1, 0},
        {0, -2, -1, 2, 1, 0},
        {0, 2, -1, -2, 1, 0},
        {0, 4, 0, -5, 0, 1},
    };
    static constexpr T G[6][3] = {
        {T(1) / 4, 0, 0},
        {T(-1) / 6, T(-1) / 6, T(-1) / 6},
        {T(-1) / 6, T(1) / 6, T(-1) / 6},
        {T(1) / 24, T(1) / 12, T(1) / 6},
        {T(1) / 24, T(-1) / 12, T(1) / 6},
        {0, 0, 1},
    };
    static constexpr T AT[4][6] = {
        {1, 1, 1, 1, 1, 0},
        {0, 1, -1, 2, -2, 0},
        {0, 1, 1, 4, 4, 0},
        {0, 1, -1, 8, -8, 1},
    };
};

// Input transform of all tiles of the planes P [nplane][hp][wp] (zero
// padded, hp = 4 th + 2, wp = 4 tw + 2): component k of tile (ty, tx)
// of plane pl goes to V[k * kstride + col], col = pl * ntiles + ty * tw
// + tx.  Vertical pass: one loop over the padded row per (plane, tile
// row) computing all six transformed rows.  Horizontal pass: the six taps
// t1[4 tx + j] are first laid out as six arrays aligned with the columns
// (a_j[col] = t1[row][4 tx + j]), so each of the 36 outputs is one long
// loop over the plane's columns.  Scratch: t1 [th][6][wp + 2], ph [6][th * tw].
template <typename T>
void wino_input(
    const T* __restrict__ P, std::size_t nplane, std::size_t hp, std::size_t wp,
    std::size_t th, std::size_t tw, T* __restrict__ V, std::size_t kstride,
    T* __restrict__ t1, T* __restrict__ ph
) {
    using M = wino_mats<T>;
    const std::size_t ntiles = th * tw;  // columns of one plane
    const std::size_t wt = wp + 2;        // t1 row stride: two zero columns for the phase split
    T* a0 = ph;
    T* a1 = ph + ntiles;
    T* a2 = ph + 2 * ntiles;
    T* a3 = ph + 3 * ntiles;
    T* a4 = ph + 4 * ntiles;
    T* a5 = ph + 5 * ntiles;
    for (std::size_t pl = 0; pl < nplane; ++pl) {
        const T* plane = P + pl * hp * wp;
        // vertical: six transformed rows per tile row, one loop each
        for (std::size_t ty = 0; ty < th; ++ty) {
            const T* d = plane + ty * WO * wp;
            T* o = t1 + ty * WT * wt;
            for (std::size_t r = 0; r < WT; ++r) {
                T* __restrict__ orow = o + r * wt;
                const T b0 = M::BT[r][0], b1 = M::BT[r][1], b2 = M::BT[r][2];
                const T b3 = M::BT[r][3], b4 = M::BT[r][4], b5 = M::BT[r][5];
                const T* d0 = d;
                const T* d1 = d + wp;
                const T* d2 = d + 2 * wp;
                const T* d3 = d + 3 * wp;
                const T* d4 = d + 4 * wp;
                const T* d5 = d + 5 * wp;
                for (std::size_t x = 0; x < wp; ++x) {
                    orow[x] = b0 * d0[x] + b1 * d1[x] + b2 * d2[x] + b3 * d3[x]
                        + b4 * d4[x] + b5 * d5[x];
                }
                orow[wp] = T(0);
                orow[wp + 1] = T(0);
            }
        }
        // horizontal, per transformed row r: split every t1 row into the
        // four phases (a stride-4 group load), shift two of them by one
        // tile for the taps j = 4, 5, then 36 loops over the plane's columns
        for (std::size_t r = 0; r < WT; ++r) {
            for (std::size_t ty = 0; ty < th; ++ty) {
                const T* __restrict__ t = t1 + (ty * WT + r) * wt;
                T* __restrict__ p0 = a0 + ty * tw;
                T* __restrict__ p1 = a1 + ty * tw;
                T* __restrict__ p2 = a2 + ty * tw;
                T* __restrict__ p3 = a3 + ty * tw;
                T* __restrict__ p4 = a4 + ty * tw;
                T* __restrict__ p5 = a5 + ty * tw;
                for (std::size_t tx = 0; tx < tw; ++tx) {
                    p0[tx] = t[4 * tx];
                    p1[tx] = t[4 * tx + 1];
                    p2[tx] = t[4 * tx + 2];
                    p3[tx] = t[4 * tx + 3];
                }
                for (std::size_t tx = 0; tx < tw; ++tx) {
                    p4[tx] = t[4 * tx + 4];
                    p5[tx] = t[4 * tx + 5];
                }
            }
            for (std::size_t s = 0; s < WT; ++s) {
                T* __restrict__ vo = V + (r * WT + s) * kstride + pl * ntiles;
                const T b0 = M::BT[s][0], b1 = M::BT[s][1], b2 = M::BT[s][2];
                const T b3 = M::BT[s][3], b4 = M::BT[s][4], b5 = M::BT[s][5];
                for (std::size_t c = 0; c < ntiles; ++c) {
                    vo[c] = b0 * a0[c] + b1 * a1[c] + b2 * a2[c] + b3 * a3[c]
                        + b4 * a4[c] + b5 * a5[c];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// The Winograd products as small matrix products: Mk[co][col] = sum_ci
// U[ci][co] V[ci][col] for one block of CO_BLK = 4 output channels, over
// columns in tiles of two 32-byte vectors (16 floats / 8 doubles): two
// vector loads and four broadcasts feed eight multiply-adds, so the
// loop is bound by the FMA units rather than by the loads.  ncol must be
// a multiple of the tile.  upk is [ci][4].
// ---------------------------------------------------------------------
template <typename T>
void gemm_block(
    const T* __restrict__ V, std::size_t cin, std::size_t ncol,
    const T* __restrict__ upk, T* __restrict__ Mk
) {
    typedef T vec __attribute__((vector_size(32)));
    constexpr std::size_t L = 32 / sizeof(T);
    for (std::size_t col = 0; col < ncol; col += 2 * L) {
        vec a00 = vec{}, a01 = vec{}, a10 = vec{}, a11 = vec{};
        vec a20 = vec{}, a21 = vec{}, a30 = vec{}, a31 = vec{};
        const T* wk = upk;
        for (std::size_t ci = 0; ci < cin; ++ci) {
            vec x0, x1;
            __builtin_memcpy(&x0, V + ci * ncol + col, 32);
            __builtin_memcpy(&x1, V + ci * ncol + col + L, 32);
            const vec w0 = vec{} + wk[0], w1 = vec{} + wk[1];
            const vec w2 = vec{} + wk[2], w3 = vec{} + wk[3];
            a00 += w0 * x0; a01 += w0 * x1;
            a10 += w1 * x0; a11 += w1 * x1;
            a20 += w2 * x0; a21 += w2 * x1;
            a30 += w3 * x0; a31 += w3 * x1;
            wk += 4;
        }
        __builtin_memcpy(Mk + 0 * ncol + col, &a00, 32);
        __builtin_memcpy(Mk + 0 * ncol + col + L, &a01, 32);
        __builtin_memcpy(Mk + 1 * ncol + col, &a10, 32);
        __builtin_memcpy(Mk + 1 * ncol + col + L, &a11, 32);
        __builtin_memcpy(Mk + 2 * ncol + col, &a20, 32);
        __builtin_memcpy(Mk + 2 * ncol + col + L, &a21, 32);
        __builtin_memcpy(Mk + 3 * ncol + col, &a30, 32);
        __builtin_memcpy(Mk + 3 * ncol + col + L, &a31, 32);
    }
}

// Output transform: M [36][cout_pad][ncolpad] -> Y [nplane][cout][n][n]
// (plane-major, n x n valid pixels), adding bias to the value plane
// (plane index pl % ncomp == 0).  The two separable passes run over all
// columns of a channel at once; scratch: z [6][4][ncol], y [4][4][ncol].
template <typename T>
void wino_output(
    const T* __restrict__ Mk, std::size_t cout, std::size_t cout_pad,
    std::size_t ncolpad, std::size_t nplane, std::size_t ncomp, std::size_t th,
    std::size_t tw, std::size_t n, const T* __restrict__ bias, T* __restrict__ Y,
    T* __restrict__ z, T* __restrict__ y
) {
    using M = wino_mats<T>;
    const std::size_t ntiles = th * tw;
    const std::size_t ncol = nplane * ntiles;
    for (std::size_t co = 0; co < cout; ++co) {
        // z[r][j][col] = sum_s AT[j][s] M[r, s][co][col]
        for (std::size_t r = 0; r < WT; ++r) {
            const T* m0 = Mk + ((r * WT + 0) * cout_pad + co) * ncolpad;
            const T* m1 = Mk + ((r * WT + 1) * cout_pad + co) * ncolpad;
            const T* m2 = Mk + ((r * WT + 2) * cout_pad + co) * ncolpad;
            const T* m3 = Mk + ((r * WT + 3) * cout_pad + co) * ncolpad;
            const T* m4 = Mk + ((r * WT + 4) * cout_pad + co) * ncolpad;
            const T* m5 = Mk + ((r * WT + 5) * cout_pad + co) * ncolpad;
            for (std::size_t j = 0; j < WO; ++j) {
                T* zo = z + (r * WO + j) * ncol;
                const T a0 = M::AT[j][0], a1 = M::AT[j][1], a2 = M::AT[j][2];
                const T a3 = M::AT[j][3], a4 = M::AT[j][4], a5 = M::AT[j][5];
                for (std::size_t c = 0; c < ncol; ++c) {
                    zo[c] = a0 * m0[c] + a1 * m1[c] + a2 * m2[c] + a3 * m3[c]
                        + a4 * m4[c] + a5 * m5[c];
                }
            }
        }
        // y[i][j][col] = sum_r AT[i][r] z[r][j][col]
        for (std::size_t i = 0; i < WO; ++i) {
            const T a0 = M::AT[i][0], a1 = M::AT[i][1], a2 = M::AT[i][2];
            const T a3 = M::AT[i][3], a4 = M::AT[i][4], a5 = M::AT[i][5];
            for (std::size_t j = 0; j < WO; ++j) {
                T* yo = y + (i * WO + j) * ncol;
                const T* z0 = z + (0 * WO + j) * ncol;
                const T* z1 = z + (1 * WO + j) * ncol;
                const T* z2 = z + (2 * WO + j) * ncol;
                const T* z3 = z + (3 * WO + j) * ncol;
                const T* z4 = z + (4 * WO + j) * ncol;
                const T* z5 = z + (5 * WO + j) * ncol;
                for (std::size_t c = 0; c < ncol; ++c) {
                    yo[c] = a0 * z0[c] + a1 * z1[c] + a2 * z2[c] + a3 * z3[c]
                        + a4 * z4[c] + a5 * z5[c];
                }
            }
        }
        // scatter the 4x4 tiles into the output planes: the four j
        // streams of a tile row interleave into one output row
        const bool whole = (n % WO == 0);
        for (std::size_t pl = 0; pl < nplane; ++pl) {
            const T add = (bias != nullptr && (pl % ncomp) == 0) ? bias[co] : T(0);
            T* out = Y + (pl * cout + co) * n * n;
            for (std::size_t ty = 0; ty < th; ++ty) {
                const std::size_t col0 = pl * ntiles + ty * tw;
                for (std::size_t i = 0; i < WO; ++i) {
                    const std::size_t oy = ty * WO + i;
                    if (oy >= n) break;
                    T* orow = out + oy * n;
                    const T* y0 = y + (i * WO + 0) * ncol + col0;
                    const T* y1 = y + (i * WO + 1) * ncol + col0;
                    const T* y2 = y + (i * WO + 2) * ncol + col0;
                    const T* y3 = y + (i * WO + 3) * ncol + col0;
                    if (whole) {
                        for (std::size_t tx = 0; tx < tw; ++tx) {
                            orow[4 * tx] = y0[tx] + add;
                            orow[4 * tx + 1] = y1[tx] + add;
                            orow[4 * tx + 2] = y2[tx] + add;
                            orow[4 * tx + 3] = y3[tx] + add;
                        }
                    } else {
                        const T* ys[4] = {y0, y1, y2, y3};
                        for (std::size_t j = 0; j < WO; ++j) {
                            const std::size_t tx_max = std::min(tw, (n - j + WO - 1) / WO);
                            for (std::size_t tx = 0; tx < tx_max; ++tx) {
                                orow[tx * WO + j] = ys[j][tx] + add;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Fused block epilogue: per-pixel channel layer norm with affine weights,
// GELU, plus res * R.  Y, R, OUT are [ncomp][C][npix] (component-major);
// the tangent planes follow the chain rule of the value plane's
// statistics: with m, inv the mean and 1 / sqrt(var + eps) over the
// channels at a pixel, mk the mean of tangent k and
// dvar_k = 2 mean((v - m)(t_k - mk)),
//   xn = (v - m) inv,  xn_k = (t_k - mk) inv + (v - m) (-1/2) inv^3 dvar_k,
//   y = xn w + b,      y_k = xn_k w,
//   out = gelu(y) + res R,   out_k = gelu'(y) y_k + res R_k.
// Scratch: stat [(2 ncomp + 1) x npix]: mean, inv, (mk, dinv_k) pairs, gelu'.
// ---------------------------------------------------------------------

inline float d4_exp_neg(float x) {
    x = x > -87.0f ? x : -87.0f;
    const float nf = (float)(int)(x * 1.44269504f + (x < 0.0f ? -0.5f : 0.5f));
    const float r = x - nf * 0.69314718f;
    float p = 1.0f / 720.0f;
    p = p * r + 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    const int e = ((int)nf + 127) << 23;
    float scale;
    __builtin_memcpy(&scale, &e, 4);
    return p * scale;
}

// exact-erf GELU through Abramowitz-Stegun 7.1.26 (|erf error| < 1.5e-7)
inline void d4_gelu(float v, float& f, float& df) {
    const float x = (v < 0.0f ? -v : v) * 0.70710678f;
    const float t = 1.0f / (1.0f + 0.3275911f * x);
    const float poly = t * (0.254829592f + t * (-0.284496736f
        + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
    const float e = d4_exp_neg(-x * x);
    const float erfc_abs = poly * e;
    const float cdf = (v >= 0.0f) ? 1.0f - 0.5f * erfc_abs : 0.5f * erfc_abs;
    const float pdf = e * 0.39894228f;
    f = v * cdf;
    df = cdf + v * pdf;
}

inline void d4_gelu(double v, double& f, double& df) {
    const double cdf = 0.5 * std::erfc(-v * M_SQRT1_2);
    const double pdf = std::exp(-0.5 * v * v) * 0.3989422804014327;
    f = v * cdf;
    df = cdf + v * pdf;
}

template <typename T>
void block_epilogue(
    const T* __restrict__ Y, const T* __restrict__ R, T* __restrict__ OUT,
    std::size_t ncomp, std::size_t C, std::size_t npix, const T* __restrict__ nw,
    const T* __restrict__ nb, T eps, T res, T* __restrict__ stat
) {
    // one pass over the channels: S1 = sum v, S2 = sum v^2, and per tangent
    // T_k = sum t_k, U_k = sum v t_k; then mean = S1 / C, var = S2 / C -
    // mean^2, m_k = T_k / C and sum (v - mean)(t_k - m_k) = U_k - C mean m_k
    T* mean = stat;             // S1, then the mean
    T* inv = stat + npix;       // S2, then 1 / sqrt(var + eps)
    const T invc = T(1) / static_cast<T>(C);
    for (std::size_t p = 0; p < npix; ++p) {
        mean[p] = T(0);
        inv[p] = T(0);
    }
    for (std::size_t k = 1; k < ncomp; ++k) {
        T* mk = stat + 2 * k * npix;
        T* dk = stat + (2 * k + 1) * npix;
        for (std::size_t p = 0; p < npix; ++p) {
            mk[p] = T(0);
            dk[p] = T(0);
        }
    }
    for (std::size_t c = 0; c < C; ++c) {
        const T* v = Y + c * npix;
        for (std::size_t p = 0; p < npix; ++p) {
            mean[p] += v[p];
            inv[p] += v[p] * v[p];
        }
        for (std::size_t k = 1; k < ncomp; ++k) {
            const T* t = Y + (k * C + c) * npix;
            T* mk = stat + 2 * k * npix;
            T* dk = stat + (2 * k + 1) * npix;
            for (std::size_t p = 0; p < npix; ++p) {
                mk[p] += t[p];
                dk[p] += v[p] * t[p];
            }
        }
    }
    for (std::size_t p = 0; p < npix; ++p) {
        const T m = mean[p] * invc;
        const T var = inv[p] * invc - m * m;
        mean[p] = m;
        inv[p] = T(1) / std::sqrt((var > T(0) ? var : T(0)) + eps);
    }
    for (std::size_t k = 1; k < ncomp; ++k) {
        T* mk = stat + 2 * k * npix;
        T* dk = stat + (2 * k + 1) * npix;
        for (std::size_t p = 0; p < npix; ++p) {
            const T m_k = mk[p] * invc;
            mk[p] = m_k;
            // dinv = -1/2 inv^3 dvar, dvar = 2 (U_k - C mean m_k) / C
            dk[p] = -inv[p] * inv[p] * inv[p] * (dk[p] * invc - mean[p] * m_k);
        }
    }
    // apply per channel: value plane, then the tangent planes
    T* dfv = stat + 2 * ncomp * npix;
    for (std::size_t c = 0; c < C; ++c) {
        const T w = nw[c], b = nb[c];
        const T* v = Y + c * npix;
        const T* rv = R + c * npix;
        T* o = OUT + c * npix;
        for (std::size_t p = 0; p < npix; ++p) {
            const T y = (v[p] - mean[p]) * inv[p] * w + b;
            T f, df;
            d4_gelu(y, f, df);
            o[p] = f + res * rv[p];
            dfv[p] = df * w;
        }
        for (std::size_t k = 1; k < ncomp; ++k) {
            const T* mk = stat + 2 * k * npix;
            const T* dk = stat + (2 * k + 1) * npix;
            const T* t = Y + (k * C + c) * npix;
            const T* rt = R + (k * C + c) * npix;
            T* ot = OUT + (k * C + c) * npix;
            for (std::size_t p = 0; p < npix; ++p) {
                const T xn_k = (t[p] - mk[p]) * inv[p] + (v[p] - mean[p]) * dk[p];
                ot[p] = dfv[p] * xn_k + res * rt[p];
            }
        }
    }
}

// ---------------------------------------------------------------------
// Inverse orbit element applied to F [nplane][n][n] into G (a plain
// permutation copy: row reversal, column reversal, transposes), then the
// sign-weighted accumulation acc1 += sa G, acc2 += sb G as two
// contiguous loops.
// ---------------------------------------------------------------------
template <typename T>
void inverse_orbit_accumulate(
    const T* __restrict__ F, std::size_t nplane, std::size_t n, int o,
    T sa, T sb, T* __restrict__ G, T* __restrict__ acc1, T* __restrict__ acc2
) {
    const int k = o % 4;
    const bool flip = o >= 4;
    const std::size_t npix = n * n;
    for (std::size_t pl = 0; pl < nplane; ++pl) {
        const T* src = F + pl * npix;
        T* dst = G + pl * npix;
        for (std::size_t i = 0; i < n; ++i) {
            // row i of F (after the optional row flip) came from canonical (y, x)
            const std::size_t ii = flip ? n - 1 - i : i;
            const T* srow = src + i * n;
            if (k == 0) {
                T* drow = dst + ii * n;
                for (std::size_t j = 0; j < n; ++j) drow[j] = srow[j];
            } else if (k == 2) {
                T* drow = dst + (n - 1 - ii) * n;
                for (std::size_t j = 0; j < n; ++j) drow[n - 1 - j] = srow[j];
            } else if (k == 1) {
                // canonical (y, x) = (j, n - 1 - ii): a column of G
                for (std::size_t j = 0; j < n; ++j) dst[j * n + (n - 1 - ii)] = srow[j];
            } else {
                // canonical (y, x) = (n - 1 - j, ii)
                for (std::size_t j = 0; j < n; ++j) dst[(n - 1 - j) * n + ii] = srow[j];
            }
        }
    }
    const std::size_t total = nplane * npix;
    for (std::size_t i = 0; i < total; ++i) {
        acc1[i] += sa * G[i];
        acc2[i] += sb * G[i];
    }
}
