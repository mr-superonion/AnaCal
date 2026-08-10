#ifndef ANACAL_PSFMODEL_H
#define ANACAL_PSFMODEL_H

#include "stdafx.h"

// Native (GIL-free) re-implementation of the LSST DM coadd-PSF
// evaluation chain: CoaddPsf -> WarpedPsf -> single-visit PSF model.
//
// The module is split into two layers so new single-visit models (e.g.
// PIFF) can be added without touching the warping/coadding machinery:
//
//   1. SingleVisitPsf (abstract) + PsfexModel: evaluate one visit's PSF
//      model at a position IN THE VISIT PIXEL FRAME, reproducing
//      meas_extensions_psfex PsfexPsf::_doComputeImage (float32
//      internals, PSFEx vignet_resample Lanczos-5 with the truncated
//      PI constant).
//   2. CoaddPsfModel: per-component visit->coadd warping (WarpedPsf:
//      affine linearization, afw lanczos warp with kernel cache
//      quantization) and the weighted combination of CoaddPsf.
//
// The visit<->coadd WCS pair is NOT evaluated here: the loader (python,
// under the GIL) fits a 2-D polynomial mapping coadd->visit per
// component once per patch, and evaluation uses that polynomial plus
// its analytic Jacobian.  Everything in this header is plain C++ with
// no Python API calls, so it is safe under a released GIL.

namespace anacal {
namespace psfmodel {

// mask_value sentinel for "no valid PSF model at this position in some
// band": such sources are kept in the catalog but always skipped by
// the measurement, independent of any mask_value_max threshold.
constexpr int psf_invalid_mask_value = 414;

// A PSF postage stamp: row-major (ny, nx) doubles plus the PARENT
// coordinate of the first pixel (the afw xy0 convention).
struct Stamp {
    std::vector<double> data;
    int nx = 0;
    int ny = 0;
    int x0 = 0;
    int y0 = 0;

    double sum() const {
        double s = 0.0;
        for (double v : data) s += v;
        return s;
    }
};

// 2x2 linear part + translation; enough to mirror lsst::geom
// AffineTransform for the WarpedPsf math.
struct Affine {
    // (u, v) = (m00 x + m01 y + tx, m10 x + m11 y + ty)
    double m00 = 1.0, m01 = 0.0, m10 = 0.0, m11 = 1.0;
    double tx = 0.0, ty = 0.0;

    inline void apply(double x, double y, double& u, double& v) const {
        u = m00 * x + m01 * y + tx;
        v = m10 * x + m11 * y + ty;
    }

    inline double det() const { return m00 * m11 - m01 * m10; }

    Affine inverse() const {
        const double d = det();
        if (d == 0.0) {
            throw std::runtime_error("psfmodel Error: singular affine");
        }
        Affine r;
        r.m00 = m11 / d;
        r.m01 = -m01 / d;
        r.m10 = -m10 / d;
        r.m11 = m00 / d;
        r.tx = -(r.m00 * tx + r.m01 * ty);
        r.ty = -(r.m10 * tx + r.m11 * ty);
        return r;
    }

    double norm_inf() const {
        return std::max(
            std::abs(m00) + std::abs(m01), std::abs(m10) + std::abs(m11)
        );
    }
};

// ---------------------------------------------------------------------
// Layer 1: single-visit models
// ---------------------------------------------------------------------

// PSFEx interpolation function (vignet.h INTERPF): Lanczos-5 with the
// TRUNCATED pi literal psfex uses -- required to match DM bit levels.
inline double psfex_interpf(double x) {
    constexpr double PI_PSFEX = 3.1415926535898;
    constexpr double INTERPFAC = 5.0;
    if (x < 1e-5 && x > -1e-5) return 1.0;
    if (x > INTERPFAC || x < -INTERPFAC) return 0.0;
    return std::sin(PI_PSFEX * x) * std::sin(PI_PSFEX / INTERPFAC * x) /
           (PI_PSFEX * PI_PSFEX / INTERPFAC * x * x);
}

// One direction of PSFEx vignet_resample: per-output-pixel 10-tap mask,
// renormalized per pixel, with the edge-truncation and skip rules of
// the original (reverse-engineered against DM and verified to float
// precision).  step2 = 1/pixstep; stepi is always 1.0 in DM usage.
struct VignetAxis {
    int ixs2 = 0;              // first output pixel written
    int n2 = 0;                // number of output pixels written
    std::vector<int> start;    // per output pixel: first input tap index
    std::vector<int> ntap;     // per output pixel: number of taps
    std::vector<float> mask;   // ntap values per output pixel (packed 10)

    // PSFEx vignet.c works in float throughout (positions, offsets and
    // masks) -- the float arithmetic here is required to match DM.
    VignetAxis(int w1, int w2, float d, float step2) {
        constexpr int interpw = 10;  // INTERPW 9 -> 10 taps at stepi=1
        constexpr int hmw = interpw / 2 - 1;
        const int mx1 = w1 / 2;
        const int mx2 = w2 / 2;
        float xs1 = mx1 + d - mx2 * step2;
        int ix2 = 0;
        if (xs1 < 0.0f) {
            const int dix2 = static_cast<int>(1.0f - xs1 / step2);
            ix2 += dix2;
            xs1 += dix2 * step2;
        }
        int nx2 = static_cast<int>((w1 - 1 - xs1) / step2 + 1.0f);
        if (nx2 > w2 - ix2) nx2 = w2 - ix2;
        if (nx2 < 0) nx2 = 0;
        ixs2 = ix2;
        n2 = nx2;
        start.resize(n2);
        ntap.resize(n2);
        mask.assign(static_cast<std::size_t>(n2) * interpw, 0.0f);
        for (int j = 0; j < n2; ++j) {
            const float x1 = xs1 + j * step2;
            const int ix1 = static_cast<int>(x1);
            int ix = ix1 - hmw;
            float dxm = ix1 - x1 - hmw;
            int n = interpw;
            if (ix < 0) {
                n += ix;
                dxm -= ix;
                ix = 0;
            }
            if (n > w1 - ix) n = w1 - ix;
            start[j] = ix;
            ntap[j] = n;
            float norm = 0.0f;
            float* mrow = &mask[static_cast<std::size_t>(j) * interpw];
            for (int k = 0; k < n; ++k) {
                mrow[k] = static_cast<float>(
                    psfex_interpf(static_cast<double>(dxm + k))
                );
                norm += mrow[k];
            }
            if (norm > 0.0f) {
                for (int k = 0; k < n; ++k) mrow[k] /= norm;
            }
        }
    }
};

inline void vignet_resample(
    const std::vector<float>& in, int w1, int h1,
    std::vector<float>& out, int w2, int h2,
    float dx, float dy, float step2
) {
    out.assign(static_cast<std::size_t>(w2) * h2, 0.0f);
    const VignetAxis ax(w1, w2, dx, step2);
    const VignetAxis ay(h1, h2, dy, step2);
    // Separable: x-resample rows into an intermediate (h1 x n2) float
    // buffer, then y-resample columns -- mirroring the original
    // two-pass float arithmetic.
    std::vector<float> tmp(static_cast<std::size_t>(h1) * ax.n2, 0.0f);
    for (int y = 0; y < h1; ++y) {
        const float* row = &in[static_cast<std::size_t>(y) * w1];
        for (int j = 0; j < ax.n2; ++j) {
            const float* m = &ax.mask[static_cast<std::size_t>(j) * 10];
            const int s = ax.start[j];
            float acc = 0.0f;
            for (int k = 0; k < ax.ntap[j]; ++k) acc += m[k] * row[s + k];
            tmp[static_cast<std::size_t>(y) * ax.n2 + j] = acc;
        }
    }
    for (int j2 = 0; j2 < ay.n2; ++j2) {
        const float* m = &ay.mask[static_cast<std::size_t>(j2) * 10];
        const int s = ay.start[j2];
        float* orow =
            &out[static_cast<std::size_t>(ay.ixs2 + j2) * w2 + ax.ixs2];
        for (int i = 0; i < ax.n2; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < ay.ntap[j2]; ++k) {
                acc += m[k] * tmp[static_cast<std::size_t>(s + k) * ax.n2 + i];
            }
            orow[i] = acc;
        }
    }
}

// Interface every single-visit PSF model implements: evaluate the
// PSF image at a position in the VISIT pixel frame, with the model's
// own sub-pixel placement (the DM ``computeImage`` semantics -- this is
// what WarpedPsf calls on the inner PSF).
class SingleVisitPsf {
public:
    virtual ~SingleVisitPsf() = default;
    virtual Stamp compute_image(double px, double py) const = 0;
    virtual Stamp compute_kernel_image(double px, double py) const = 0;
};

// PSFEx model: basis components on an oversampled grid, coefficients
// from a position polynomial (meas_extensions_psfex PsfexPsf).
class PsfexModel : public SingleVisitPsf {
public:
    int w, h, nbasis;
    std::vector<float> comp;              // basis-major, row-major (y*w+x)
    float pixstep;
    int degree;                           // single polynomial group
    std::array<double, 2> ctx_offset;
    std::array<double, 2> ctx_scale;

    PsfexModel(
        int w_, int h_, int nbasis_,
        std::vector<float> comp_,
        double pixstep_,
        int degree_,
        std::array<double, 2> ctx_offset_,
        std::array<double, 2> ctx_scale_
    ) : w(w_), h(h_), nbasis(nbasis_), comp(std::move(comp_)),
        pixstep(static_cast<float>(pixstep_)), degree(degree_),
        ctx_offset(ctx_offset_), ctx_scale(ctx_scale_)
    {
        if (static_cast<std::size_t>(w) * h * nbasis != comp.size()) {
            throw std::runtime_error(
                "psfmodel Error: comp size does not match (w, h, nbasis)"
            );
        }
        const int ncoeff = (degree + 1) * (degree + 2) / 2;
        if (ncoeff != nbasis) {
            throw std::runtime_error(
                "psfmodel Error: nbasis does not match a 2-D polynomial "
                "of the given degree (only ndim=2, ngroup=1 PSFEx models "
                "are supported)"
            );
        }
    }

    // poly_func basis for ndim=2, ngroup=1:
    //   for j in 0..d: for i in 0..d-j: x^i * y^j
    std::vector<double> basis_at(double px, double py) const {
        const double x = (px - ctx_offset[0]) / ctx_scale[0];
        const double y = (py - ctx_offset[1]) / ctx_scale[1];
        std::vector<double> b;
        b.reserve(nbasis);
        double yj = 1.0;
        for (int j = 0; j <= degree; ++j) {
            double xi = 1.0;
            for (int i = 0; i <= degree - j; ++i) {
                b.push_back(xi * yj);
                xi *= x;
            }
            yj *= y;
        }
        return b;
    }

    Stamp compute_image(double px, double py) const override {
        return impl(px, py, px, py);
    }

    Stamp compute_kernel_image(double px, double py) const override {
        return impl(px, py, 0.0, 0.0);
    }

private:
    // PsfexPsf::_doComputeImage(position, center): float32 throughout,
    // matching DM exactly.
    Stamp impl(double px, double py, double cx, double cy) const {
        const std::vector<double> basis = basis_at(px, py);
        const std::size_t npix = static_cast<std::size_t>(w) * h;
        std::vector<float> fullres(npix, 0.0f);
        for (int t = 0; t < nbasis; ++t) {
            const float fac = static_cast<float>(basis[t]);
            const float* ppc = &comp[static_cast<std::size_t>(t) * npix];
            for (std::size_t k = 0; k < npix; ++k) {
                fullres[k] += fac * ppc[k];
            }
        }
        const float vigstep = 1.0f / pixstep;
        // DM: ``float dx = center[0] - static_cast<int>(center[0])`` --
        // the subtraction happens in DOUBLE and only the result narrows
        // to float.  Narrowing cx first would misplace non-representable
        // fractions (0.3, 0.7, ...) by ~5e-5 px.
        float dx = static_cast<float>(cx - static_cast<int>(cx));
        float dy = static_cast<float>(cy - static_cast<int>(cy));
        if (dx > 0.5f) dx -= 1.0f;
        if (dy > 0.5f) dy -= 1.0f;
        int sampleW = static_cast<int>(w * pixstep);
        int sampleH = static_cast<int>(h * pixstep);
        if (sampleW % 2 == 0) sampleW += 1;
        if (sampleH % 2 == 0) sampleH += 1;
        Stamp st;
        st.nx = sampleW;
        st.ny = sampleH;
        st.x0 = static_cast<int>(cx - dx + 0.5) - sampleW / 2;
        st.y0 = static_cast<int>(cy - dy + 0.5) - sampleH / 2;
        std::vector<float> sampled;
        vignet_resample(
            fullres, w, h, sampled, sampleW, sampleH,
            -dx * vigstep,
            -dy * vigstep,
            vigstep
        );
        float sum = 0.0f;
        for (float v : sampled) sum += v;
        st.data.resize(sampled.size());
        for (std::size_t k = 0; k < sampled.size(); ++k) {
            st.data[k] = static_cast<double>(sampled[k] / sum);
        }
        return st;
    }
};

// PIFF model (meas_extensions_piff PiffPsf with a piff PixelGrid +
// BasisPolynomial): the kernel image is simply the interpolated
// parameter grid itself (q . basis(u, v), renormalized); the sub-pixel
// image resamples that grid with a separable galsim Lanczos
// (conserve_dc) kernel.  All arithmetic is double precision.
class PiffModel : public SingleVisitPsf {
public:
    int nmodel;                   // model grid size N (odd)
    int stamp;                    // output stamp size (= N for DP1)
    double coord_scale;           // u = coord_scale * x_detector
    std::vector<double> q;        // (N*N, nterm), row = iy*N + ix
    std::vector<int> term_i;      // u exponent per term
    std::vector<int> term_j;      // v exponent per term
    int lanczos_n;                // galsim Lanczos order (11 for DP1)
    std::vector<double> dc_coeff; // conserve_dc corrections c_m

    PiffModel(
        int nmodel_, int stamp_, double coord_scale_,
        std::vector<double> q_,
        std::vector<int> term_i_, std::vector<int> term_j_,
        int lanczos_n_, std::vector<double> dc_coeff_
    ) : nmodel(nmodel_), stamp(stamp_), coord_scale(coord_scale_),
        q(std::move(q_)), term_i(std::move(term_i_)),
        term_j(std::move(term_j_)), lanczos_n(lanczos_n_),
        dc_coeff(std::move(dc_coeff_))
    {
        if (nmodel % 2 == 0 || stamp % 2 == 0) {
            throw std::runtime_error(
                "psfmodel Error: PIFF model/stamp sizes must be odd"
            );
        }
        // The output stamp IS the interpolated parameter grid (see
        // compute_image), so a DM stamp size that differs from the
        // model grid -- a PSF fitted with samplingSize != 1 -- would
        // silently return the wrong-sized image.  Reject it instead;
        // the loader also refuses oversampled models.
        if (stamp != nmodel) {
            throw std::runtime_error(
                "psfmodel Error: PIFF stamp size must equal the model "
                "grid size (oversampled models are not supported)"
            );
        }
        if (q.size() != static_cast<std::size_t>(nmodel) * nmodel *
                            term_i.size() ||
            term_i.size() != term_j.size()) {
            throw std::runtime_error(
                "psfmodel Error: PIFF q/terms shapes are inconsistent"
            );
        }
    }

    // galsim Lanczos(n, conserve_dc=True): sinc(x) sinc(x/n) times a
    // periodic flux-conservation correction (coefficients supplied by
    // the loader, solved from galsim's own xval).
    inline double kfun(double x) const {
        const double ax = std::abs(x);
        if (ax >= lanczos_n) return 0.0;
        constexpr double PI_AFW = 3.141592653589793238462643383279506;
        auto sinc = [](double t) {
            if (std::fabs(t) < 1e-15) return 1.0;
            const double pt = 3.141592653589793238462643383279506 * t;
            return std::sin(pt) / pt;
        };
        double val = sinc(x) * sinc(x / lanczos_n);
        double corr = 1.0;
        for (std::size_t m = 1; m <= dc_coeff.size(); ++m) {
            corr += dc_coeff[m - 1] *
                    (1.0 - std::cos(2.0 * PI_AFW * m * x));
        }
        return val * corr;
    }

    // params(u, v) = q . basis, normalized to unit sum, reshaped (N, N)
    std::vector<double> params_at(double px, double py) const {
        const double u = coord_scale * px;
        const double v = coord_scale * py;
        const std::size_t nterm = term_i.size();
        int maxi = 0, maxj = 0;
        for (std::size_t k = 0; k < nterm; ++k) {
            maxi = std::max(maxi, term_i[k]);
            maxj = std::max(maxj, term_j[k]);
        }
        std::vector<double> pu(maxi + 1, 1.0), pv(maxj + 1, 1.0);
        for (int a = 1; a <= maxi; ++a) pu[a] = pu[a - 1] * u;
        for (int b = 1; b <= maxj; ++b) pv[b] = pv[b - 1] * v;
        std::vector<double> basis(nterm);
        for (std::size_t k = 0; k < nterm; ++k) {
            basis[k] = pu[term_i[k]] * pv[term_j[k]];
        }
        const std::size_t npix =
            static_cast<std::size_t>(nmodel) * nmodel;
        std::vector<double> params(npix, 0.0);
        for (std::size_t r = 0; r < npix; ++r) {
            const double* qr = &q[r * nterm];
            double acc = 0.0;
            for (std::size_t k = 0; k < nterm; ++k) {
                acc += qr[k] * basis[k];
            }
            params[r] = acc;
        }
        double s = 0.0;
        for (double vv : params) s += vv;
        for (double& vv : params) vv /= s;
        return params;
    }

    Stamp compute_kernel_image(double px, double py) const override {
        Stamp st;
        st.nx = st.ny = nmodel;
        st.x0 = st.y0 = -(nmodel / 2);
        st.data = params_at(px, py);
        // params are already normalized; the final /sum of PiffPsf is a
        // no-op here but kept for exactness
        const double s = st.sum();
        for (double& vv : st.data) vv /= s;
        return st;
    }

    Stamp compute_image(double px, double py) const override {
        // galsim stamp centre: ceil(p - 0.5); DM labels the bbox with
        // round-half-up -- both reproduced deliberately (they differ at
        // exact half-integers, a known DM quirk).
        const double xcen = std::ceil(px - 0.5);
        const double ycen = std::ceil(py - 0.5);
        const double dx = px - xcen;
        const double dy = py - ycen;
        const int half = nmodel / 2;
        Stamp st;
        st.nx = st.ny = nmodel;
        st.x0 = static_cast<int>(std::floor(px + 0.5)) - half;
        st.y0 = static_cast<int>(std::floor(py + 0.5)) - half;
        const std::vector<double> params = params_at(px, py);
        if (dx == 0.0 && dy == 0.0) {
            st.data = params;
        } else {
            // separable resample: out[jy][jx] =
            //   sum_iy K(iy - jy + dy) sum_ix K(ix - jx + dx) P[iy][ix]
            const int n = nmodel;
            std::vector<double> wx(
                static_cast<std::size_t>(n) * n
            ), wy(static_cast<std::size_t>(n) * n);
            for (int jc = 0; jc < n; ++jc) {
                for (int ic = 0; ic < n; ++ic) {
                    wx[static_cast<std::size_t>(jc) * n + ic] =
                        kfun((ic - jc) + dx);
                    wy[static_cast<std::size_t>(jc) * n + ic] =
                        kfun((ic - jc) + dy);
                }
            }
            std::vector<double> tmp(
                static_cast<std::size_t>(n) * n, 0.0
            );
            for (int iy = 0; iy < n; ++iy) {
                const double* prow =
                    &params[static_cast<std::size_t>(iy) * n];
                for (int jx = 0; jx < n; ++jx) {
                    const double* w =
                        &wx[static_cast<std::size_t>(jx) * n];
                    double acc = 0.0;
                    for (int ix = 0; ix < n; ++ix) {
                        acc += w[ix] * prow[ix];
                    }
                    tmp[static_cast<std::size_t>(iy) * n + jx] = acc;
                }
            }
            st.data.assign(static_cast<std::size_t>(n) * n, 0.0);
            for (int jy = 0; jy < n; ++jy) {
                const double* w =
                    &wy[static_cast<std::size_t>(jy) * n];
                for (int jx = 0; jx < n; ++jx) {
                    double acc = 0.0;
                    for (int iy = 0; iy < n; ++iy) {
                        acc += w[iy] *
                               tmp[static_cast<std::size_t>(iy) * n + jx];
                    }
                    st.data[static_cast<std::size_t>(jy) * n + jx] = acc;
                }
            }
        }
        const double s = st.sum();
        for (double& vv : st.data) vv /= s;
        return st;
    }
};

// ---------------------------------------------------------------------
// Layer 2: afw warping + CoaddPsf combination
// ---------------------------------------------------------------------

// afw LanczosFunction1: full double pi, NOT truncated at |x| > n.
inline double lanczos_afw(double x, int n) {
    constexpr double PI_AFW = 3.141592653589793238462643383279506;
    const double a1 = x * PI_AFW;
    const double a2 = a1 / n;
    if (std::fabs(a1) > 1.0e-5) {
        return std::sin(a1) * std::sin(a2) / (a1 * a2);
    }
    return 1.0;
}

// afw LanczosWarpingKernel of a given order with the SeparableKernel
// tap cache: cache row i holds taps for fractional offset
// (i + 0.5)/cacheSize; lookup truncates frac*cacheSize.  cacheSize = 0
// evaluates exactly.
class WarpKernel {
public:
    int order;      // lanczos order n
    int size;       // 2n
    int ctr;        // (size - 1) / 2
    int cache_size;
    std::vector<double> cache;      // cache_size rows of `size` taps
    std::vector<double> cache_sum;  // per-row tap sum

    WarpKernel(int order_, int cache_size_)
        : order(order_), size(2 * order_), ctr(order_ - 1),
          cache_size(cache_size_)
    {
        if (cache_size > 0) {
            cache.resize(static_cast<std::size_t>(cache_size) * size);
            cache_sum.resize(cache_size);
            for (int i = 0; i < cache_size; ++i) {
                const double f = (i + 0.5) / static_cast<double>(cache_size);
                double s = 0.0;
                for (int j = 0; j < size; ++j) {
                    const double v = lanczos_afw((j - ctr) - f, order);
                    cache[static_cast<std::size_t>(i) * size + j] = v;
                    s += v;
                }
                cache_sum[i] = s;
            }
        }
    }

    // Fill taps for fractional offset f in [0, 1); returns the tap sum.
    double taps(double f, double* out) const {
        if (cache_size > 0) {
            int idx = static_cast<int>(f * cache_size);
            if (idx >= cache_size) idx = cache_size - 1;
            const double* row = &cache[static_cast<std::size_t>(idx) * size];
            for (int j = 0; j < size; ++j) out[j] = row[j];
            return cache_sum[idx];
        }
        double s = 0.0;
        for (int j = 0; j < size; ++j) {
            out[j] = lanczos_afw((j - ctr) - f, order);
            s += out[j];
        }
        return s;
    }
};

// WarpedPsf::warpAffine + afw warpImage (interpLength = 0), including
// the destination-bbox rule (odd, centered on 0) and the trailing
// normalization of WarpedPsf::doComputeKernelImage.
inline Stamp warp_affine(
    const Stamp& src, const Affine& src_to_dest, const WarpKernel& kernel
) {
    if (src_to_dest.norm_inf() > 200.0) {
        throw std::runtime_error(
            "psfmodel Error: unexpectedly large transform in warp_affine"
        );
    }
    const int pad = std::max(kernel.ctr, kernel.size - kernel.ctr);

    // computeBBoxFromTransform on the UNPADDED source bbox
    const double cx0 = src.x0 - 0.5;
    const double cy0 = src.y0 - 0.5;
    const double cx1 = src.x0 + src.nx - 0.5;
    const double cy1 = src.y0 + src.ny - 0.5;
    double ox0 = std::numeric_limits<double>::infinity();
    double oy0 = ox0;
    double ox1 = -ox0;
    double oy1 = -ox0;
    const double xs[4] = {cx0, cx1, cx0, cx1};
    const double ys[4] = {cy0, cy0, cy1, cy1};
    for (int c = 0; c < 4; ++c) {
        double u, v;
        src_to_dest.apply(xs[c], ys[c], u, v);
        ox0 = std::min(ox0, u);
        ox1 = std::max(ox1, u);
        oy0 = std::min(oy0, v);
        oy1 = std::max(oy1, v);
    }
    const int hx = static_cast<int>(std::floor(0.5 * (ox1 - ox0)));
    const int hy = static_cast<int>(std::floor(0.5 * (oy1 - oy0)));

    Stamp dst;
    dst.nx = 2 * hx + 1;
    dst.ny = 2 * hy + 1;
    dst.x0 = -hx;
    dst.y0 = -hy;
    dst.data.assign(static_cast<std::size_t>(dst.nx) * dst.ny, 0.0);

    // zero-padded source (xy0 shifted so PARENT coords are unchanged)
    const int wp = src.nx + 2 * pad;
    const int hp = src.ny + 2 * pad;
    const int px0 = src.x0 - pad;
    const int py0 = src.y0 - pad;
    std::vector<double> padded(static_cast<std::size_t>(wp) * hp, 0.0);
    for (int y = 0; y < src.ny; ++y) {
        std::copy(
            src.data.begin() + static_cast<std::size_t>(y) * src.nx,
            src.data.begin() + static_cast<std::size_t>(y + 1) * src.nx,
            padded.begin() + static_cast<std::size_t>(y + pad) * wp + pad
        );
    }

    const Affine dest_to_src = src_to_dest.inverse();
    const double rel_area = std::abs(dest_to_src.det());

    // srcGoodBBox = shrinkBBox of the padded LOCAL bbox
    const int good_x0 = kernel.ctr;
    const int good_y0 = kernel.ctr;
    const int good_x1 = kernel.ctr + (wp - kernel.size);   // inclusive max
    const int good_y1 = kernel.ctr + (hp - kernel.size);

    std::vector<double> kx(kernel.size), ky(kernel.size);
    double total = 0.0;
    for (int dy = 0; dy < dst.ny; ++dy) {
        const double yparent = dst.y0 + dy;
        for (int dxp = 0; dxp < dst.nx; ++dxp) {
            const double xparent = dst.x0 + dxp;
            double sx, sy;
            dest_to_src.apply(xparent, yparent, sx, sy);
            // local (padded) coordinates
            const double lx = sx - px0;
            const double ly = sy - py0;
            int i0 = static_cast<int>(std::floor(lx + 0.5));
            int j0 = static_cast<int>(std::floor(ly + 0.5));
            double fx = lx - i0;
            double fy = ly - j0;
            if (fx < 0.0) { fx += 1.0; --i0; }
            if (fy < 0.0) { fy += 1.0; --j0; }
            if (i0 < good_x0 || i0 > good_x1 ||
                j0 < good_y0 || j0 > good_y1) {
                continue;  // padValue = 0
            }
            const double sumx = kernel.taps(fx, kx.data());
            const double sumy = kernel.taps(fy, ky.data());
            const int sx0 = i0 - kernel.ctr;
            const int sy0 = j0 - kernel.ctr;
            double acc = 0.0;
            for (int j = 0; j < kernel.size; ++j) {
                const double* row =
                    &padded[static_cast<std::size_t>(sy0 + j) * wp + sx0];
                double rowacc = 0.0;
                for (int i = 0; i < kernel.size; ++i) {
                    rowacc += row[i] * kx[i];
                }
                acc += rowacc * ky[j];
            }
            const double val = acc * rel_area / (sumx * sumy);
            dst.data[static_cast<std::size_t>(dy) * dst.nx + dxp] = val;
            total += val;
        }
    }
    if (total == 0.0) {
        throw std::runtime_error(
            "psfmodel Error: warped psf image has sum 0"
        );
    }
    for (double& v : dst.data) v /= total;
    return dst;
}

// afw offsetImage(im, dx, dy, "lanczos5", buffer=5) as used by
// Psf::recenterKernelImage: |dx|,|dy| <= 0.5, fresh (uncached) kernel,
// ctr bumped by +1 on axes with negative shift, per-axis normalized
// taps, zero buffer.
inline Stamp offset_image_lanczos5(const Stamp& in, double dx, double dy) {
    constexpr int n = 5;
    constexpr int size = 2 * n;
    constexpr int buffer = 5;
    const double dkx = -dx;
    const double dky = -dy;
    int ctrx = n - 1;
    int ctry = n - 1;
    if (dkx < 0.0) ctrx += 1;
    if (dky < 0.0) ctry += 1;
    double kx[size], ky[size];
    double sx = 0.0, sy = 0.0;
    for (int i = 0; i < size; ++i) {
        kx[i] = lanczos_afw((i - ctrx) - dkx, n);
        sx += kx[i];
        ky[i] = lanczos_afw((i - ctry) - dky, n);
        sy += ky[i];
    }
    for (int i = 0; i < size; ++i) {
        kx[i] /= sx;
        ky[i] /= sy;
    }
    const int wp = in.nx + 2 * buffer;
    const int hp = in.ny + 2 * buffer;
    std::vector<double> padded(static_cast<std::size_t>(wp) * hp, 0.0);
    for (int y = 0; y < in.ny; ++y) {
        std::copy(
            in.data.begin() + static_cast<std::size_t>(y) * in.nx,
            in.data.begin() + static_cast<std::size_t>(y + 1) * in.nx,
            padded.begin() + static_cast<std::size_t>(y + buffer) * wp
                + buffer
        );
    }
    Stamp out;
    out.nx = in.nx;
    out.ny = in.ny;
    out.x0 = in.x0;
    out.y0 = in.y0;
    out.data.assign(in.data.size(), 0.0);
    // convolve (afw orientation): dest(x, y) =
    //   sum_ij padded(x + i - ctrx, y + j - ctry) * kx[i] * ky[j],
    // evaluated at the crop region [buffer, buffer + n) of the padded
    // frame; taps reaching outside the padded image cannot happen for
    // buffer = 5 and ctr in {4, 5}.
    for (int y = 0; y < in.ny; ++y) {
        const int yp = y + buffer;
        for (int x = 0; x < in.nx; ++x) {
            const int xp = x + buffer;
            double acc = 0.0;
            for (int j = 0; j < size; ++j) {
                const double* row = &padded[
                    static_cast<std::size_t>(yp + j - ctry) * wp
                    + (xp - ctrx)
                ];
                double rowacc = 0.0;
                for (int i = 0; i < size; ++i) {
                    rowacc += row[i] * kx[i];
                }
                acc += rowacc * ky[j];
            }
            out.data[static_cast<std::size_t>(y) * in.nx + x] = acc;
        }
    }
    return out;
}

// 2-D tensor-polynomial mapping coadd -> visit pixels, fitted by the
// python loader; evaluation gives the mapped point and the analytic
// Jacobian (the affine linearization DM computes from the WCS pair).
struct MappingPoly {
    int order = 0;
    double cx = 0.0, cy = 0.0, sx = 1.0, sy = 1.0;  // normalization
    std::vector<double> cu;  // (order+1)^2, cu[a*(order+1)+b]: x'^a y'^b
    std::vector<double> cv;

    void eval(
        double x, double y,
        double& u, double& v,
        double& dudx, double& dudy, double& dvdx, double& dvdy
    ) const {
        const int m = order + 1;
        const double xn = (x - cx) / sx;
        const double yn = (y - cy) / sy;
        std::vector<double> xp(m, 1.0), yp(m, 1.0);
        for (int a = 1; a < m; ++a) xp[a] = xp[a - 1] * xn;
        for (int b = 1; b < m; ++b) yp[b] = yp[b - 1] * yn;
        u = v = dudx = dudy = dvdx = dvdy = 0.0;
        for (int a = 0; a < m; ++a) {
            for (int b = 0; b < m; ++b) {
                const double cu_ab = cu[static_cast<std::size_t>(a) * m + b];
                const double cv_ab = cv[static_cast<std::size_t>(a) * m + b];
                const double xy = xp[a] * yp[b];
                u += cu_ab * xy;
                v += cv_ab * xy;
                if (a > 0) {
                    const double t = a * xp[a - 1] * yp[b];
                    dudx += cu_ab * t;
                    dvdx += cv_ab * t;
                }
                if (b > 0) {
                    const double t = b * xp[a] * yp[b - 1];
                    dudy += cu_ab * t;
                    dvdy += cv_ab * t;
                }
            }
        }
        dudx /= sx;
        dvdx /= sx;
        dudy /= sy;
        dvdy /= sy;
    }
};

// Interface for any per-source PSF the ForceTask can draw from inside
// its GIL-released loop.  This is the ONLY per-source PSF mechanism:
// pre-drawn Python stamp stacks are not supported.
class PerSourcePsf {
public:
    virtual ~PerSourcePsf() = default;
    // False where no valid PSF model exists (the source is then flagged
    // with psf_invalid_mask_value and skipped in that band).
    virtual bool contains(double x, double y) const = 0;
    virtual Stamp compute_image(double x, double y) const = 0;
};

// Per-source PSF from a coarse grid of pre-computed stamps (nearest
// cell, no interpolation) -- the survey-agnostic GridPsf (Euclid MER
// PSF, pre-sampled models) evaluated natively.  Mirrors the xlens
// GridPsf.draw indexing exactly: cell (i, j) covers
// x in [j*dx, (j+1)*dx), y in [i*dy, (i+1)*dy); outside positions are
// clamped to the nearest edge cell.
class GridPsfModel : public PerSourcePsf {
public:
    int ngy, ngx, npix;
    double dx, dy;
    std::vector<double> stamps;  // (ngy, ngx, npix, npix) row-major

    GridPsfModel(
        int ngy_, int ngx_, int npix_, double dx_, double dy_,
        std::vector<double> stamps_
    ) : ngy(ngy_), ngx(ngx_), npix(npix_), dx(dx_), dy(dy_),
        stamps(std::move(stamps_))
    {
        if (stamps.size() != static_cast<std::size_t>(ngy) * ngx *
                                 npix * npix) {
            throw std::runtime_error(
                "psfmodel Error: GridPsfModel stamp array shape mismatch"
            );
        }
    }

    bool contains(double, double) const override { return true; }

    Stamp compute_image(double x, double y) const override {
        int j = static_cast<int>(std::floor(x / dx));
        int i = static_cast<int>(std::floor(y / dy));
        j = std::min(std::max(j, 0), ngx - 1);
        i = std::min(std::max(i, 0), ngy - 1);
        Stamp st;
        st.nx = st.ny = npix;
        st.x0 = st.y0 = -(npix / 2);
        const double* src = &stamps[
            (static_cast<std::size_t>(i) * ngx + j) *
            (static_cast<std::size_t>(npix) * npix)
        ];
        st.data.assign(
            src, src + static_cast<std::size_t>(npix) * npix
        );
        return st;
    }
};

// One CoaddPsf input record.
struct CoaddComponent {
    std::shared_ptr<SingleVisitPsf> model;
    double weight = 1.0;
    // Box2D of the visit bbox (outer float box: min - 0.5 .. max + 0.5)
    double bx0 = 0.0, bx1 = 0.0, by0 = 0.0, by1 = 0.0;
    // validPolygon vertices in the visit frame (empty: bbox only)
    std::vector<double> poly_x;
    std::vector<double> poly_y;
    MappingPoly map;  // coadd pixels -> visit pixels

    // DM ExposureRecord::contains: bbox AND (if present) polygon.
    // Polygon::contains is boost::geometry::within (boundary exclusive);
    // even-odd ray casting reproduces it away from edges.
    bool contains(double u, double v) const {
        if (!(u >= bx0 && u <= bx1 && v >= by0 && v <= by1)) return false;
        const std::size_t nv = poly_x.size();
        if (nv < 3) return true;
        bool inside = false;
        for (std::size_t i = 0, j = nv - 1; i < nv; j = i++) {
            if (((poly_y[i] > v) != (poly_y[j] > v)) &&
                (u < (poly_x[j] - poly_x[i]) * (v - poly_y[i]) /
                         (poly_y[j] - poly_y[i]) + poly_x[i])) {
                inside = !inside;
            }
        }
        return inside;
    }
};

// The coadd-level model: CoaddPsf::doComputeKernelImage /
// doComputeImage without any Python or WCS calls.
//
// THREADING: evaluation is const and allocation-free, so one model may
// be shared by any number of threads -- but ``components`` must be
// fully built (add_component) BEFORE that sharing begins: evaluation
// holds references into the vector with the GIL released, so a
// concurrent push_back that reallocates would dangle them.
class CoaddPsfModel : public PerSourcePsf {
public:
    WarpKernel kernel;
    std::vector<CoaddComponent> components;

    CoaddPsfModel(int lanczos_order, int cache_size)
        : kernel(lanczos_order, cache_size) {}

    // True when at least one coadd input covers the point -- the same
    // rule evaluation uses, so this predicts exactly where
    // compute_*_image would throw (DM: InvalidPsfError "no input
    // images at that point").
    bool contains(double x, double y) const override {
        for (const CoaddComponent& c : components) {
            double u, v, dudx, dudy, dvdx, dvdy;
            c.map.eval(x, y, u, v, dudx, dudy, dvdx, dvdy);
            if (c.contains(u, v)) return true;
        }
        return false;
    }

    Stamp compute_kernel_image(double x, double y) const {
        std::vector<Stamp> imgs;
        std::vector<double> weights;
        double wsum = 0.0;
        for (const CoaddComponent& c : components) {
            double u, v, dudx, dudy, dvdx, dvdy;
            c.map.eval(x, y, u, v, dudx, dudy, dvdx, dvdy);
            if (!c.contains(u, v)) continue;
            // A = linearization of coadd->visit at (x, y)
            Affine A;
            A.m00 = dudx;
            A.m01 = dudy;
            A.m10 = dvdx;
            A.m11 = dvdy;
            A.tx = u - dudx * x - dudy * y;
            A.ty = v - dvdx * x - dvdy * y;
            const double rx = std::round(x);
            const double ry = std::round(y);
            double su, sv;
            A.apply(rx, ry, su, sv);              // src_position
            Affine S = A.inverse();               // visit -> coadd
            S.tx -= rx;                           // recentre on (0, 0)
            S.ty -= ry;
            const Stamp inner = c.model->compute_image(su, sv);
            imgs.push_back(warp_affine(inner, S, kernel));
            weights.push_back(c.weight);
            wsum += c.weight;
        }
        if (imgs.empty()) {
            throw std::runtime_error(
                "psfmodel Error: no coadd inputs at the requested point"
            );
        }
        // union bbox
        int x0 = imgs[0].x0, y0 = imgs[0].y0;
        int x1 = imgs[0].x0 + imgs[0].nx, y1 = imgs[0].y0 + imgs[0].ny;
        for (const Stamp& s : imgs) {
            x0 = std::min(x0, s.x0);
            y0 = std::min(y0, s.y0);
            x1 = std::max(x1, s.x0 + s.nx);
            y1 = std::max(y1, s.y0 + s.ny);
        }
        Stamp out;
        out.x0 = x0;
        out.y0 = y0;
        out.nx = x1 - x0;
        out.ny = y1 - y0;
        out.data.assign(static_cast<std::size_t>(out.nx) * out.ny, 0.0);
        for (std::size_t k = 0; k < imgs.size(); ++k) {
            const Stamp& s = imgs[k];
            const double scale = weights[k] / s.sum();
            for (int yy = 0; yy < s.ny; ++yy) {
                double* orow = &out.data[
                    static_cast<std::size_t>(s.y0 - y0 + yy) * out.nx
                    + (s.x0 - x0)
                ];
                const double* srow =
                    &s.data[static_cast<std::size_t>(yy) * s.nx];
                for (int xx = 0; xx < s.nx; ++xx) {
                    orow[xx] += scale * srow[xx];
                }
            }
        }
        for (double& vv : out.data) vv /= wsum;
        return out;
    }

    // Psf::computeImage = kernel image recentered onto the requested
    // (fractional) position; NOT renormalized.
    Stamp compute_image(double x, double y) const override {
        Stamp im = compute_kernel_image(x, y);
        const double ix = std::floor(x + 0.5);
        const double iy = std::floor(y + 0.5);
        const double rx = x - ix;
        const double ry = y - iy;
        if (rx != 0.0 || ry != 0.0) {
            im = offset_image_lanczos5(im, rx, ry);
        }
        im.x0 += static_cast<int>(ix);
        im.y0 += static_cast<int>(iy);
        return im;
    }
};

// Centre crop / zero-pad a stamp to (tny, tnx), with the exact xlens
// ``resize_array`` conventions (crop start (in-out)//2; rows pad
// bottom-first, columns pad left-first).  ``out`` must hold tny*tnx
// doubles; pure C++, safe under a released GIL.
inline void resize_stamp_to(
    const Stamp& st, int tny, int tnx, double* out
) {
    std::fill(out, out + static_cast<std::size_t>(tny) * tnx, 0.0);
    int src_y0 = 0, dst_y0 = 0, ny = st.ny;
    if (st.ny > tny) {
        src_y0 = (st.ny - tny) / 2;
        ny = tny;
    } else if (st.ny < tny) {
        const int pad = tny - st.ny;
        dst_y0 = pad - pad / 2;
    }
    int src_x0 = 0, dst_x0 = 0, nx = st.nx;
    if (st.nx > tnx) {
        src_x0 = (st.nx - tnx) / 2;
        nx = tnx;
    } else if (st.nx < tnx) {
        const int pad = tnx - st.nx;
        dst_x0 = pad - pad / 2;
    }
    for (int y = 0; y < ny; ++y) {
        const double* srow = &st.data[
            static_cast<std::size_t>(src_y0 + y) * st.nx + src_x0
        ];
        double* drow =
            &out[static_cast<std::size_t>(dst_y0 + y) * tnx + dst_x0];
        std::copy(srow, srow + nx, drow);
    }
}

inline void resize_stamp_to(const Stamp& st, int npix, double* out) {
    resize_stamp_to(st, npix, npix, out);
}

void pyExportPsfModel(py::module_& m);

}  // namespace psfmodel
}  // namespace anacal

#endif  // ANACAL_PSFMODEL_H
