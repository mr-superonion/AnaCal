#ifndef ANACAL_RENOISE_H
#define ANACAL_RENOISE_H

#include "image.h"

// Minimal-variance renoising ("Option 2"): the spectral quantities that
// the shear-response correction needs, following "Minimal-variance
// renoising" (anacal_renoise/main.tex), Sections 4.1, 4.5 and 4.6.
//
// Everything here lives on the rfft half-plane of a SQUARE n x n grid of
// pixel scale ``scale``: index = j * (n/2 + 1) + i, with k1 = i dk for
// i in [0, n/2] and k2 = m_j dk, m_j = j for j < n/2 and j - n otherwise,
// dk = 2 pi / (n scale).  This is the layout of Image::draw_f_vec and of
// numpy.fft.rfft2.
//
// Two layers:
//   * NoiseSpectrum -- the image-noise power spectrum S(k) of ONE survey
//     patch and band, its 90-degree rotation S(Rk) and its shear
//     derivatives; built once per patch from the noise correlation
//     function (or a constant for white noise).
//   * RenoiseKernels -- per PSF: the deconvolved spectrum S^d = S / |p|^2,
//     its 90-degree-symmetrized version S^d_sym (a smooth maximum of
//     S^d(k) and S^d(Rk)), the filler filter g that raises the image
//     noise to S^d_sym, the amplitude w = sqrt(c S^d_sym / S^d) of the
//     added noise and the assigned-response filters kappa_1, kappa_2 =
//     -sqrt(S^d_sym / S^d) a_sym / (2 sqrt c), where a_sym is the shear
//     derivative of ln S^d_sym.  The smoothing Gaussian h cancels from
//     all of them, so one set serves detection and measurement.
//
// All k-derivatives are exact on the grid: d/dk_j F[f] = -i F[x_j f], so
// the derivative of S comes from the transforms of x_j C and that of
// |p|^2 from the transforms of x_j p; no finite differences.

namespace anacal {
namespace renoise {

// Fraction of the maximum below which a noise spectrum is floored (a
// measured correlation function is noisy at high k) and its log-derivative
// set to zero; the PSF floor is min_deconv_ratio of image.h.
inline constexpr double spectrum_floor_ratio = 1e-8;

inline int
half_width(int n) { return n / 2 + 1; }

// Which noise correction the kernels / Setup describe: Option 1 (the
// rotated pure-noise copy, "rotate") or Option 2 (the spectral
// renoising, "spectral").  Both carry the noise spectrum of the patch, so
// the variances of the renoised image are summed over the actual total
// noise spectrum in either mode.
enum class Mode { rotate = 1, spectral = 2 };

// (R a)(k) = a(R k) = a(-k2, k1) for a REAL, EVEN array stored on the
// half-plane: a(-k2, k1) = a(k2, -k1) by evenness, which brings the
// first argument back to k1 >= 0 whichever sign k2 has.  The Nyquist
// row and column (m = n/2) have no exact partner (+-k_N alias onto one
// sample); they map onto themselves and are zeroed by the callers.
inline std::vector<double>
rotate90_even(const std::vector<double>& a, int n) {
    const int kxl = half_width(n);
    if (static_cast<int>(a.size()) != n * kxl) {
        throw std::invalid_argument("rotate90_even: array / grid mismatch");
    }
    std::vector<double> out(a.size());
    for (int j = 0; j < n; ++j) {
        const int m = (j < n / 2) ? j : j - n;      // k2 index
        for (int i = 0; i <= n / 2; ++i) {           // k1 index >= 0
            int i2, m2;
            if (m <= 0) {
                i2 = -m;                             // k1' = -k2 >= 0
                m2 = i;                              // k2' = k1
            } else {
                i2 = m;                              // evenness: (k2, -k1)
                m2 = -i;
            }
            if (i2 > n / 2) i2 = n / 2;              // m = -n/2 (Nyquist)
            const int j2 = (m2 + n) % n;
            out[static_cast<std::size_t>(j) * kxl + i] =
                a[static_cast<std::size_t>(j2) * kxl + i2];
        }
    }
    return out;
}

struct NoiseSpectrum {
    int n = 0;
    double scale = 0.0;
    bool white = true;
    double variance = 0.0;       // per-pixel variance of the raw noise
    // true after symmetrize_spectrum: S is the spectrum of the image noise
    // WITH the spin-2 filler (patch PSF), S_raw that of the pure-noise
    // realizations
    bool symmetrized = false;
    // half-plane arrays
    std::vector<double> S;      // S(k)
    std::vector<double> S_R;    // S(R k)
    std::vector<double> dS1;    // (k1 d/dk1 - k2 d/dk2) S
    std::vector<double> dS2;    // (k1 d/dk2 + k2 d/dk1) S
    std::vector<double> S_raw;    // raw noise spectrum (= S until symmetrized)
    std::vector<double> S_raw_R;  // S_raw(R k)
    std::vector<double> g;        // filler filter (set by symmetrize_spectrum)
};

// Forward r2c transform of a real array placed with its centre pixel
// (arr // 2) at the FFT origin of an n x n grid (zero padded or cropped).
inline std::vector<std::complex<double>>
centred_fft(const double* arr, int arr_ny, int arr_nx, int n, double scale) {
    ImageLease lease(n, n, scale, true);
    Image& img = lease.get();
    img.set_r_raw(arr, arr_ny, arr_nx, true);
    img.fft();
    return img.draw_f_vec();
}

// NoiseSpectrum of one patch and band.  ``corr`` is the noise
// correlation function C(x) (lag 0 at pixel (ny//2, nx//2), C(0) = the
// per-pixel variance); without it the noise is white with the given
// variance.  The grid must be larger than the support of C so that the
// lags do not alias.
inline NoiseSpectrum
make_noise_spectrum(
    int n,
    double scale,
    double variance,
    const std::optional<py::array_t<double>>& corr = std::nullopt
) {
    if (n < 4 || n % 2 != 0) {
        throw std::invalid_argument("make_noise_spectrum: n must be even");
    }
    NoiseSpectrum ns;
    ns.n = n;
    ns.scale = scale;
    const int kxl = half_width(n);
    const std::size_t npix = static_cast<std::size_t>(n) * kxl;
    ns.variance = variance;
    if (!corr.has_value()) {
        ns.white = true;
        ns.S.assign(npix, variance);
        ns.S_R.assign(npix, variance);
        ns.dS1.assign(npix, 0.0);
        ns.dS2.assign(npix, 0.0);
        ns.S_raw = ns.S;
        ns.S_raw_R = ns.S_R;
        return ns;
    }
    ns.white = false;
    auto c = corr->unchecked<2>();
    const int cny = static_cast<int>(c.shape(0));
    const int cnx = static_cast<int>(c.shape(1));
    if (cny > n || cnx > n) {
        throw std::invalid_argument(
            "make_noise_spectrum: correlation function larger than the grid"
        );
    }
    // the correlation function gives the SHAPE of the spectrum; its
    // amplitude is set by ``variance`` (C(0) is rescaled to it), so a
    // unit-normalised correlation (xlens convention) and one carrying the
    // per-pixel variance give the same spectrum
    const double c_centre = c(cny / 2, cnx / 2);
    if (!(c_centre > 0.0)) {
        throw std::invalid_argument(
            "make_noise_spectrum: the correlation function must be positive "
            "at zero lag (pixel (ny // 2, nx // 2))"
        );
    }
    const double amp = variance / c_centre;
    std::vector<double> c0(static_cast<std::size_t>(cny) * cnx);
    std::vector<double> c1(c0.size()), c2(c0.size());
    for (int j = 0; j < cny; ++j) {
        const double y = (j - cny / 2) * scale;
        for (int i = 0; i < cnx; ++i) {
            const double x = (i - cnx / 2) * scale;
            const std::size_t idx = static_cast<std::size_t>(j) * cnx + i;
            c0[idx] = amp * c(j, i);
            c1[idx] = x * c0[idx];
            c2[idx] = y * c0[idx];
        }
    }
    const auto F0 = centred_fft(c0.data(), cny, cnx, n, scale);
    const auto F1 = centred_fft(c1.data(), cny, cnx, n, scale);
    const auto F2 = centred_fft(c2.data(), cny, cnx, n, scale);
    ns.S.resize(npix);
    ns.dS1.resize(npix);
    ns.dS2.resize(npix);
    double smax = 0.0;
    for (std::size_t idx = 0; idx < npix; ++idx) {
        ns.S[idx] = F0[idx].real();
        smax = std::max(smax, ns.S[idx]);
    }
    const double floor = spectrum_floor_ratio * smax;
    const double dk = 2.0 * M_PI / (n * scale);
    for (int j = 0; j < n; ++j) {
        const double k2 = ((j < n / 2) ? j : j - n) * dk;
        for (int i = 0; i < kxl; ++i) {
            const double k1 = i * dk;
            const std::size_t idx = static_cast<std::size_t>(j) * kxl + i;
            if (ns.S[idx] < floor) {
                ns.S[idx] = floor;
                ns.dS1[idx] = 0.0;
                ns.dS2[idx] = 0.0;
            } else {
                // d S / d k_j = -i F[x_j C] = Im F[x_j C]  (F[x_j C] is
                // purely imaginary for a real, even C)
                const double d1 = F1[idx].imag();
                const double d2 = F2[idx].imag();
                ns.dS1[idx] = k1 * d1 - k2 * d2;
                ns.dS2[idx] = k1 * d2 + k2 * d1;
            }
        }
    }
    ns.S_R = rotate90_even(ns.S, n);
    ns.S_raw = ns.S;
    ns.S_raw_R = ns.S_R;
    return ns;
}

struct RenoiseKernels {
    int n = 0;
    double scale = 0.0, klim = 0.0, c = 0.0, delta = 0.0;
    // half-plane arrays; zero outside |k| < klim and on the Nyquist row
    // and column (where S^d_sym is set to S^d)
    std::vector<double> Sd;        // S / |p_eff|^2
    std::vector<double> Sd_sym;    // smooth max of S^d(k), S^d(R k)
    std::vector<double> a_sym1;    // (k1 d1 - k2 d2) ln S^d_sym
    std::vector<double> a_sym2;    // (k1 d2 + k2 d1) ln S^d_sym
    std::vector<double> g;         // filler filter, sqrt(S^d_sym / S^d - 1)
    std::vector<double> w;         // sqrt(c S^d_sym / S^d)
    std::vector<double> kappa1;    // -sqrt(S^d_sym / S^d) a_sym1 / (2 sqrt c)
    std::vector<double> kappa2;    // -sqrt(S^d_sym / S^d) a_sym2 / (2 sqrt c)
    std::vector<double> ap1;       // (k1 d1 - k2 d2) ln |p|^2
    std::vector<double> ap2;       // (k1 d2 + k2 d1) ln |p|^2
    std::vector<double> Sd0;       // variance / |p_eff|^2 (white reference)
    // deconvolved spectrum of the Option 1 pure-noise term, measured
    // with the rotated kernel h / p(R k): S_src(k) / |p(R k)|^2, with
    // S_src the spectrum the pure-noise image was drawn from
    std::vector<double> Sd_rot;
    Mode mode = Mode::spectral;
    // the pure-noise image has spectrum S_raw(R k) (xlens draws it from
    // the 90-degree-rotated correlation) instead of S_raw(k): w and
    // kappa_j carry the matching sqrt(S_raw / S_raw_R)
    bool noise_rotated = false;
};

// Per-PSF kernels.  ``psf`` is centred on pixel (npsf // 2, npsf // 2)
// (AnaCal's PSF convention) and normalised to unit sum; ``klim`` is the
// band limit of the deconvolution (prepare_data_cell uses 3.1 / scale);
// ``c`` the added-noise fraction; ``delta`` the smoothing of the maximum
// (equation fill of the draft).
template <typename PsfAt>
inline RenoiseKernels
make_renoise_kernels_impl(
    const NoiseSpectrum& ns,
    PsfAt p_r,          // p_r(j, i): PSF pixel (row j, column i)
    int pny,
    int pnx,
    double klim,
    double c,
    double delta,
    // smooth 90-degree maximum of S^d (the filler's spectrum); false
    // uses the spectrum as given (a spectrum already symmetrized with
    // the patch PSF, see symmetrize_spectrum)
    bool symmetrize=true,
    bool noise_rotated=false,
    Mode mode=Mode::spectral
) {
    if (!(c > 0.0)) {
        throw std::invalid_argument("make_renoise_kernels: c must be > 0");
    }
    if (!(delta >= 0.0)) {
        throw std::invalid_argument("make_renoise_kernels: delta must be >= 0");
    }
    const int n = ns.n;
    const double scale = ns.scale;
    const int kxl = half_width(n);
    const std::size_t npix = static_cast<std::size_t>(n) * kxl;

    std::vector<double> p0(static_cast<std::size_t>(pny) * pnx);
    std::vector<double> p1(p0.size()), p2(p0.size());
    for (int j = 0; j < pny; ++j) {
        const double y = (j - pny / 2) * scale;
        for (int i = 0; i < pnx; ++i) {
            const double x = (i - pnx / 2) * scale;
            const std::size_t idx = static_cast<std::size_t>(j) * pnx + i;
            p0[idx] = p_r(j, i);
            p1[idx] = x * p_r(j, i);
            p2[idx] = y * p_r(j, i);
        }
    }
    const auto P0 = centred_fft(p0.data(), pny, pnx, n, scale);
    const auto P1 = centred_fft(p1.data(), pny, pnx, n, scale);
    const auto P2 = centred_fft(p2.data(), pny, pnx, n, scale);
    if (std::abs(P0[0].imag()) > 1e-10) {
        throw std::runtime_error(
            "make_renoise_kernels: PSF is not real in configuration space"
        );
    }
    const double pfloor = min_deconv_ratio * P0[0].real();

    RenoiseKernels rk;
    rk.n = n;
    rk.scale = scale;
    rk.klim = klim;
    rk.c = c;
    rk.delta = delta;
    rk.noise_rotated = noise_rotated;
    rk.mode = mode;
    rk.Sd.assign(npix, 0.0);
    rk.Sd0.assign(npix, 0.0);
    rk.Sd_rot.assign(npix, 0.0);
    rk.ap1.assign(npix, 0.0);
    rk.ap2.assign(npix, 0.0);
    std::vector<double> dSd1(npix, 0.0), dSd2(npix, 0.0), pabs2(npix, 0.0);
    std::vector<char> inband(npix, 0);

    const double dk = 2.0 * M_PI / (n * scale);
    const double klim2 = klim * klim;
    for (int j = 0; j < n; ++j) {
        const double k2 = ((j < n / 2) ? j : j - n) * dk;
        for (int i = 0; i < kxl; ++i) {
            const double k1 = i * dk;
            const std::size_t idx = static_cast<std::size_t>(j) * kxl + i;
            // the Nyquist row / column have no rotation partner
            if (k1 * k1 + k2 * k2 > klim2 || j == n / 2 || i == n / 2) {
                continue;
            }
            inband[idx] = 1;
            const std::complex<double>& pv = P0[idx];
            const double pabs = std::abs(pv);
            double ap1 = 0.0, ap2 = 0.0, p2abs;
            if (pabs < pfloor) {
                p2abs = pfloor * pfloor;          // floored: ln|p|^2 constant
            } else {
                p2abs = pabs * pabs;
                // d ln|p|^2 / dk_j = 2 Re[(dp/dk_j) / p] = 2 Im[F[x_j p] / p]
                const std::complex<double> q1 = P1[idx] / pv;
                const std::complex<double> q2 = P2[idx] / pv;
                ap1 = 2.0 * (k1 * q1.imag() - k2 * q2.imag());
                ap2 = 2.0 * (k1 * q2.imag() + k2 * q1.imag());
            }
            const double S = ns.S[idx];
            rk.Sd[idx] = S / p2abs;
            rk.Sd0[idx] = ns.variance / p2abs;
            rk.ap1[idx] = ap1;
            rk.ap2[idx] = ap2;
            pabs2[idx] = p2abs;
            // (k d) S^d = S^d [(k d) ln S - (k d) ln |p|^2]
            dSd1[idx] = rk.Sd[idx] * (ns.dS1[idx] / S - ap1);
            dSd2[idx] = rk.Sd[idx] * (ns.dS2[idx] / S - ap2);
        }
    }
    // rotated copies; the derivative of S^d(R k) under a spin-2 operator
    // is minus the rotated derivative
    const std::vector<double> Sd_R = rotate90_even(rk.Sd, n);
    const std::vector<double> dSd1_R = rotate90_even(dSd1, n);
    const std::vector<double> dSd2_R = rotate90_even(dSd2, n);

    rk.Sd_sym.assign(npix, 0.0);
    rk.a_sym1.assign(npix, 0.0);
    rk.a_sym2.assign(npix, 0.0);
    rk.g.assign(npix, 0.0);
    rk.w.assign(npix, 0.0);
    rk.kappa1.assign(npix, 0.0);
    rk.kappa2.assign(npix, 0.0);
    const std::vector<double>& S_src = noise_rotated ? ns.S_raw_R : ns.S_raw;
    const std::vector<double> pabs2_R = rotate90_even(pabs2, n);
    for (std::size_t idx = 0; idx < npix; ++idx) {
        if (!inband[idx]) {
            rk.Sd_sym[idx] = rk.Sd[idx];
            continue;
        }
        if (pabs2_R[idx] > 0.0) {
            rk.Sd_rot[idx] = S_src[idx] / pabs2_R[idx];
        }
        const double s = rk.Sd[idx];
        const double sR = Sd_R[idx];
        double ssym = s;
        if (symmetrize && (s > 0.0) && (sR > 0.0)) {
            // smooth maximum of S^d(k) and S^d(R k), with the chain rule
            // on both spin-2 operators (the rotation partner outside the
            // band leaves the mode unsymmetrized)
            const double d = s - sR;
            const double sig = s + sR;
            const double D = std::sqrt(d * d + delta * delta * sig * sig);
            ssym = 0.5 * (sig + D);
            const double dS1 = dSd1[idx], dS1R = -dSd1_R[idx];
            const double dS2 = dSd2[idx], dS2R = -dSd2_R[idx];
            const double inv = (D > 0.0) ? 1.0 / D : 0.0;
            rk.a_sym1[idx] = 0.5 / ssym * (
                dS1 + dS1R + (d * (dS1 - dS1R) + delta * delta * sig * (dS1 + dS1R)) * inv
            );
            rk.a_sym2[idx] = 0.5 / ssym * (
                dS2 + dS2R + (d * (dS2 - dS2R) + delta * delta * sig * (dS2 + dS2R)) * inv
            );
        } else if (s > 0.0) {
            // the spectrum as given: a = (k d) ln S^d
            rk.a_sym1[idx] = dSd1[idx] / s;
            rk.a_sym2[idx] = dSd2[idx] / s;
        }
        rk.Sd_sym[idx] = ssym;
        if (!(s > 0.0)) {
            continue;
        }
        rk.g[idx] = std::sqrt(std::max(ssym / s - 1.0, 0.0));
        // added noise w * noise with spectrum c S^d_sym |p|^2, from a
        // pure-noise realization of spectrum S_src
        const double src = S_src[idx];
        if (src > 0.0) {
            rk.w[idx] = std::sqrt(c * ssym * pabs2[idx] / src);
            rk.kappa1[idx] = -rk.a_sym1[idx] * rk.w[idx] / (2.0 * c);
            rk.kappa2[idx] = -rk.a_sym2[idx] * rk.w[idx] / (2.0 * c);
        }
    }
    return rk;
}

// Band ``band`` of a 2-D PSF stamp or a (nband, npsf, npsf) stack, read
// through unchecked accessors (no numpy allocation: GIL-free).
inline RenoiseKernels
make_renoise_kernels(
    const NoiseSpectrum& ns,
    const py::array_t<double>& psf_stack,
    ssize_t band,
    double klim,
    double c,
    double delta,
    bool symmetrize=true,
    bool noise_rotated=false,
    Mode mode=Mode::spectral
) {
    if (psf_stack.ndim() == 2) {
        if (band != 0) {
            throw std::invalid_argument(
                "make_renoise_kernels: band index out of range for a 2-D PSF"
            );
        }
        auto r = psf_stack.unchecked<2>();
        return make_renoise_kernels_impl(
            ns, [&](int j, int i) { return r(j, i); },
            static_cast<int>(r.shape(0)), static_cast<int>(r.shape(1)),
            klim, c, delta, symmetrize, noise_rotated, mode
        );
    }
    if (psf_stack.ndim() == 3) {
        if (band < 0 || band >= psf_stack.shape(0)) {
            throw std::invalid_argument(
                "make_renoise_kernels: band index out of range"
            );
        }
        auto r = psf_stack.unchecked<3>();
        return make_renoise_kernels_impl(
            ns, [&](int j, int i) { return r(band, j, i); },
            static_cast<int>(r.shape(1)), static_cast<int>(r.shape(2)),
            klim, c, delta, symmetrize, noise_rotated, mode
        );
    }
    throw std::invalid_argument(
        "make_renoise_kernels: psf must be 2-D or a 3-D band stack"
    );
}

inline RenoiseKernels
make_renoise_kernels(
    const NoiseSpectrum& ns,
    const py::array_t<double>& psf,
    double klim,
    double c,
    double delta,
    bool symmetrize=true,
    bool noise_rotated=false,
    Mode mode=Mode::spectral
) {
    if (psf.ndim() != 2) {
        throw std::invalid_argument("make_renoise_kernels: psf must be 2-D");
    }
    return make_renoise_kernels(
        ns, psf, 0, klim, c, delta, symmetrize, noise_rotated, mode
    );
}

// The spectrum of the image noise AFTER the spin-2 filler of the patch:
// S_s = S S^d_sym / S^d for the patch PSF (the filler g * noise'' with
// g = sqrt(S^d_sym / S^d - 1) is added to the image once per patch and
// band, see xlens).  The shear derivatives follow from the chain rule,
// (k d) ln S_s = a_sym + (k d) ln |p|^2.  S_raw keeps the spectrum of the
// pure-noise realizations.  ``g`` is the filler filter on this grid.
inline NoiseSpectrum
symmetrize_spectrum(
    const NoiseSpectrum& ns,
    const py::array_t<double>& psf,
    double klim,
    double delta
) {
    if (ns.symmetrized) {
        throw std::invalid_argument(
            "symmetrize_spectrum: the spectrum is already symmetrized"
        );
    }
    const RenoiseKernels rk = make_renoise_kernels(
        ns, psf, klim, 1.0, delta, true, false
    );
    NoiseSpectrum out = ns;
    out.symmetrized = true;
    out.g = rk.g;
    const std::size_t npix = ns.S.size();
    for (std::size_t idx = 0; idx < npix; ++idx) {
        if (!(rk.Sd[idx] > 0.0)) {
            // outside the band: unchanged (the band limit removes it
            // anyway) and no derivative
            out.dS1[idx] = 0.0;
            out.dS2[idx] = 0.0;
            continue;
        }
        out.S[idx] = ns.S[idx] * rk.Sd_sym[idx] / rk.Sd[idx];
        out.dS1[idx] = out.S[idx] * (rk.a_sym1[idx] + rk.ap1[idx]);
        out.dS2[idx] = out.S[idx] * (rk.a_sym2[idx] + rk.ap2[idx]);
    }
    out.S_R = rotate90_even(out.S, ns.n);
    return out;
}


// Total deconvolved noise spectrum of the renoised image at one mode:
// Option 1 S^d + S^d_rot (image noise plus the rotated pure-noise copy),
// Option 2 (1 + c) S^d_sym (symmetrized image noise plus the added
// noise).
inline double
total_spectrum_at(const RenoiseKernels& rk, std::size_t idx) {
    if (rk.mode == Mode::rotate) {
        return rk.Sd[idx] + rk.Sd_rot[idx];
    }
    return (1.0 + rk.c) * rk.Sd_sym[idx];
}

inline std::vector<double>
total_spectrum(const RenoiseKernels& rk) {
    std::vector<double> out(rk.Sd.size());
    for (std::size_t idx = 0; idx < out.size(); ++idx) {
        out[idx] = total_spectrum_at(rk, idx);
    }
    return out;
}

// Variance of the Gaussian-smoothed (sigma_arcsec) renoised image
// (symmetrized image noise plus added noise) relative to the smoothed
// WHITE noise of the raw per-pixel variance: (1 + c) sum |h|^2 S^d_sym /
// sum |h|^2 S^d_0.  Replaces the factor 2 of the rotated-noise option in
// the Task's variance bookkeeping, whose helpers assume white noise.
inline double
variance_factor(const RenoiseKernels& rk, double sigma_arcsec) {
    const int n = rk.n;
    const int kxl = half_width(n);
    const double dk = 2.0 * M_PI / (n * rk.scale);
    const double s2 = sigma_arcsec * sigma_arcsec;
    double num = 0.0, den = 0.0;
    for (int j = 0; j < n; ++j) {
        const double k2 = ((j < n / 2) ? j : j - n) * dk;
        for (int i = 0; i < kxl; ++i) {
            const double k1 = i * dk;
            const std::size_t idx = static_cast<std::size_t>(j) * kxl + i;
            const double wgt = (i == 0 || i == n / 2) ? 1.0 : 2.0;
            const double h2 = std::exp(-(k1 * k1 + k2 * k2) * s2);
            den += wgt * h2 * rk.Sd0[idx];
            num += wgt * h2 * total_spectrum_at(rk, idx);
        }
    }
    if (!(den > 0.0)) {
        throw std::runtime_error("variance_factor: empty band");
    }
    return num / den;
}

// Variance of a Gaussian aperture flux (kernel sigma_kernel on top of the
// smoothing sigma_smooth, the flux of gaussian_flux_variance_band in
// task.h) under the total noise of Option 2, (1 + c) S^d_sym: the direct
// sum of the aperture's Fourier window over the deconvolved spectrum,
// with the normalisation of gaussian_flux_variance_band (which is the
// same sum over the white reference S^d_0 = variance / |p|^2).
inline double
gaussian_flux_variance(
    const RenoiseKernels& rk,
    double sigma_kernel,
    double sigma_smooth
) {
    const int n = rk.n;
    const int kxl = half_width(n);
    const double dk = 2.0 * M_PI / (n * rk.scale);
    const double s2 = sigma_kernel * sigma_kernel
        + 2.0 * sigma_smooth * sigma_smooth;
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        const double k2 = ((j < n / 2) ? j : j - n) * dk;
        for (int i = 0; i < kxl; ++i) {
            const double k1 = i * dk;
            const std::size_t idx = static_cast<std::size_t>(j) * kxl + i;
            const double wgt = (i == 0 || i == n / 2) ? 1.0 : 2.0;
            sum += wgt * std::exp(-(k1 * k1 + k2 * k2) * s2)
                * total_spectrum_at(rk, idx);
        }
    }
    const double sigma_pix = std::sqrt(
        sigma_kernel * sigma_kernel + sigma_smooth * sigma_smooth
    ) / rk.scale;
    const double ff = 4.0 * M_PI * sigma_pix * sigma_pix;
    const double var = sum * ff * ff
        / (static_cast<double>(n) * static_cast<double>(n));
    return (var > 0.0) ? var : 0.0;
}

// Everything the Task needs to run Option 2 on one patch: the noise
// spectrum of every band (from the patch correlation function, on the
// cell grid) and the two parameters.  Absent (nullptr) means Option 1.
struct Setup {
    Mode mode = Mode::spectral;
    double c = 0.1;
    bool noise_rotated = true;
    // one per band, on the cell grid, SYMMETRIZED with the patch PSF
    // (symmetrize_spectrum) when the image carries the filler
    std::vector<NoiseSpectrum> spectra;
};

// One kernel set per band for the PSF of one cell.
inline std::vector<RenoiseKernels>
make_cell_kernels(
    const Setup& setup,
    const py::array_t<double>& psf_stack,
    double klim
) {
    const ssize_t nband = nband_of(psf_stack);
    if (static_cast<ssize_t>(setup.spectra.size()) != nband) {
        throw std::runtime_error(
            "anacal Error: renoise setup has " +
            std::to_string(setup.spectra.size()) +
            " noise spectrum(s) for " + std::to_string(nband) + " band(s)"
        );
    }
    std::vector<RenoiseKernels> out;
    out.reserve(nband);
    for (ssize_t b = 0; b < nband; ++b) {
        // the spectrum as given: no filler is applied inside anacal
        // (Option 2 needs the spectrum of the image WITH the filler,
        // Option 1 the raw one)
        const bool want_sym = (setup.mode == Mode::spectral);
        if (setup.spectra[b].symmetrized != want_sym) {
            throw std::runtime_error(
                want_sym
                ? "anacal Error: spectral renoising needs the symmetrized "
                  "noise spectrum (symmetrize_spectrum)"
                : "anacal Error: the rotate mode needs the raw noise "
                  "spectrum (make_noise_spectrum)"
            );
        }
        out.push_back(make_renoise_kernels(
            setup.spectra[b], psf_stack, b, klim, setup.c, 0.0,
            false, setup.noise_rotated, setup.mode
        ));
    }
    return out;
}

} // namespace renoise

// Option 2 in the qnumber image.  The image already carries the spin-2
// filler (added per patch, see symmetrize_spectrum).  Fourier buffers
// (half plane, karr the deconvolution-plus-resmoothing kernel h / p of
// this band):
//   D-buffer  fd = karr F[image]
//   kn = karr F[noise]            (the one pure-noise image)
//   value     fv = fd + w kn      (added noise)
//   g_j slot += F^-1[kappa_j kn]  (assigned response)
// The pixel shear response (G1, G2, X1, X2 filters and the coordinate
// terms) acts on fd only.
inline std::vector<math::qnumber>
ImageQ::prepare_qnumber_vector_band_spectral(
    const py::array_t<pixel_t>& img_stack,
    const py::array_t<double>& psf_stack,
    ssize_t band,
    int xcen,
    int ycen,
    const py::array_t<pixel_t>& noise_stack,
    const renoise::RenoiseKernels& rk
) {
    if ((rk.n != this->nx) || (rk.n != this->ny)) {
        throw std::runtime_error(
            "ImageQ Error: renoise kernels are on a " + std::to_string(rk.n) +
            " grid but the image is " + std::to_string(this->nx) + " x " +
            std::to_string(this->ny) + " (cells must be square)"
        );
    }
    if (std::abs(rk.scale - this->scale) > 1e-10 * this->scale) {
        throw std::runtime_error(
            "ImageQ Error: renoise kernels have a different pixel scale"
        );
    }
    const std::vector<std::complex<double>> karr =
        img_obj.get_lens_kernel_vec(psf_stack, band, gauss_model, klim);
    const std::size_t nf = karr.size();

    // image -> D-buffer
    img_obj.set_r_band(img_stack, band, xcen, ycen, false);
    img_obj.fft();
    img_obj.filter(karr);
    const std::vector<std::complex<double>> fd = img_obj.draw_f_vec();

    // noise
    img_obj.set_r_band(noise_stack, band, xcen, ycen, false);
    img_obj.fft();
    img_obj.filter(karr);
    const std::vector<std::complex<double>> kn = img_obj.draw_f_vec();

    std::vector<std::complex<double>> fv(nf), fk1(nf), fk2(nf);
    for (std::size_t idx = 0; idx < nf; ++idx) {
        fv[idx] = fd[idx] + rk.w[idx] * kn[idx];
        fk1[idx] = rk.kappa1[idx] * kn[idx];
        fk2[idx] = rk.kappa2[idx] * kn[idx];
    }

    const ssize_t npix = static_cast<ssize_t>(this->ny) * this->nx;
    std::vector<math::qnumber> result(npix);
    const double nfac = img_obj.norm();
    // v
    {
        img_obj.set_f(fv);
        img_obj.ifft_raw();
        const double* tmp_r = img_obj.view_r();
        for (ssize_t idx = 0; idx < npix; ++idx) {
            result[idx].v = tmp_r[idx] * nfac;
        }
    }
    // g1 = G1 (image + filler) + assigned response
    {
        img_obj.set_f(fd);
        img_obj.filter(gauss_g1_model);
        img_obj.add_f(fk1);
        img_obj.ifft_raw();
        const double* tmp_r = img_obj.view_r();
        for (ssize_t idx = 0; idx < npix; ++idx) {
            result[idx].g1 = tmp_r[idx] * nfac;
        }
    }
    // g2
    {
        img_obj.set_f(fd);
        img_obj.filter(gauss_g2_model);
        img_obj.add_f(fk2);
        img_obj.ifft_raw();
        const double* tmp_r = img_obj.view_r();
        for (ssize_t idx = 0; idx < npix; ++idx) {
            result[idx].g2 = tmp_r[idx] * nfac;
        }
    }
    // x1 (and the coordinate terms of the shear response)
    {
        img_obj.set_f(fd);
        img_obj.filter(gauss_x1_model);
        img_obj.ifft_raw();
        const double* tmp_r = img_obj.view_r();
        for (ssize_t j = 0, idx = 0; j < this->ny; ++j) {
            const double y = (j - this->ny2) * this->scale;
            for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                const double x = (i - this->nx2) * this->scale;
                const double t = tmp_r[idx] * nfac;
                result[idx].g1 = result[idx].g1 + x * t;
                result[idx].g2 = result[idx].g2 + y * t;
                result[idx].x1 = t;
            }
        }
    }
    // x2
    {
        img_obj.set_f(fd);
        img_obj.filter(gauss_x2_model);
        img_obj.ifft_raw();
        const double* tmp_r = img_obj.view_r();
        for (ssize_t j = 0, idx = 0; j < this->ny; ++j) {
            const double y = (j - this->ny2) * this->scale;
            for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                const double x = (i - this->nx2) * this->scale;
                const double t = tmp_r[idx] * nfac;
                result[idx].g1 = result[idx].g1 - y * t;
                result[idx].g2 = result[idx].g2 + x * t;
                result[idx].x2 = t;
            }
        }
    }
    return result;
}

namespace renoise {

void pyExportRenoise(py::module_& m);

} // namespace renoise
} // namespace anacal

#endif // ANACAL_RENOISE_H
