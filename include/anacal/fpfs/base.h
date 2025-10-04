#ifndef ANACAL_FPFS_BASE_H
#define ANACAL_FPFS_BASE_H

#include "../image.h"
#include "../math.h"
#include "../mask.h"
#include "../stdafx.h"

#include <limits>
#include <vector>

namespace anacal {
    inline constexpr double fpfs_det_sigma2 = 0.04;
    inline constexpr double fpfs_cut_sigma_ratio = 1.6;

    namespace fpfs_detail {
        inline std::vector<double>
        make_rfftfreq(int n) {
            const int n_r = n / 2 + 1;
            std::vector<double> freq(n_r);
            const double factor = 2.0 * M_PI / static_cast<double>(n);
            for (int k = 0; k < n_r; ++k) {
                freq[k] = factor * static_cast<double>(k);
            }
            return freq;
        }

        inline std::vector<double>
        make_fftfreq(int n) {
            std::vector<double> freq(n);
            const double factor = 2.0 * M_PI / static_cast<double>(n);
            const int n_half = n / 2;
            for (int k = 0; k <= n_half; ++k) {
                freq[k] = factor * static_cast<double>(k);
            }
            for (int k = n_half + 1; k < n; ++k) {
                freq[k] = factor * static_cast<double>(k - n);
            }
            return freq;
        }

        inline std::complex<double>
        pow_i(int n) {
            switch (n % 4) {
                case 0: return {1.0, 0.0};
                case 1: return {0.0, 1.0};
                case 2: return {-1.0, 0.0};
                default: return {0.0, -1.0};
            }
        }

        inline size_t
        idx4(int n, int m, int y, int x, int norder, int mord, int ny, int nx) {
            return (
                (static_cast<size_t>(n) * (mord + 1) + static_cast<size_t>(m)) * ny
                + static_cast<size_t>(y)
            ) * nx + static_cast<size_t>(x);
        }

        struct ShapeletResult {
            int ny;
            int nx;
            int norder;
            std::vector<std::complex<double>> chi;
        };

        inline ShapeletResult
        compute_shapelet_modes(int npix, int norder, double sigma, double kmax) {
            const int ny = npix;
            const int nx = npix / 2 + 1;
            const int mord = norder;
            const int nmodes = (norder + 1) * (mord + 1);
            const size_t total_size = static_cast<size_t>(nmodes) * ny * nx;

            std::vector<double> xfreq = make_rfftfreq(npix);
            std::vector<double> yfreq = make_fftfreq(npix);

            std::vector<double> gauss(static_cast<size_t>(ny) * nx, 0.0);
            std::vector<double> r2_over_sigma2(static_cast<size_t>(ny) * nx, 0.0);
            std::vector<std::complex<double>> eulfunc(static_cast<size_t>(ny) * nx);

            const double kmax2 = kmax * kmax;
            const double sigma2 = sigma * sigma;
            for (int j = 0; j < ny; ++j) {
                const double ky = yfreq[j];
                for (int i = 0; i < nx; ++i) {
                    const double kx = xfreq[i];
                    const double r2 = kx * kx + ky * ky;
                    const bool inside = r2 <= kmax2;
                    const size_t idx = static_cast<size_t>(j) * nx + static_cast<size_t>(i);
                    if (inside) {
                        gauss[idx] = std::exp(-r2 / (2.0 * sigma2));
                    } else {
                        gauss[idx] = 0.0;
                    }
                    if (sigma2 > 0.0) {
                        r2_over_sigma2[idx] = r2 / sigma2;
                    } else {
                        r2_over_sigma2[idx] = 0.0;
                    }
                    if (r2 > 0.0) {
                        const double inv_r = 1.0 / std::sqrt(r2);
                        const double cos_phi = kx * inv_r;
                        const double sin_phi = ky * inv_r;
                        eulfunc[idx] = {cos_phi, sin_phi};
                    } else {
                        eulfunc[idx] = {0.0, 0.0};
                    }
                }
            }

            std::vector<double> lfunc(static_cast<size_t>(norder + 1) * (mord + 1) * ny * nx, 0.0);
            for (int m = 0; m <= mord; ++m) {
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        lfunc[idx4(0, m, j, i, norder, mord, ny, nx)] = 1.0;
                        if (norder >= 1) {
                            lfunc[idx4(1, m, j, i, norder, mord, ny, nx)] = (
                                1.0 - r2_over_sigma2[static_cast<size_t>(j) * nx + static_cast<size_t>(i)]
                                + static_cast<double>(m)
                            );
                        }
                    }
                }
            }

            for (int n = 2; n <= norder; ++n) {
                for (int m = 0; m <= mord; ++m) {
                    const double a_coeff = 2.0;
                    const double b_coeff = 1.0 + (static_cast<double>(m) - 1.0) / static_cast<double>(n);
                    for (int j = 0; j < ny; ++j) {
                        for (int i = 0; i < nx; ++i) {
                            const size_t base_idx = static_cast<size_t>(j) * nx + static_cast<size_t>(i);
                            const double r_term = (
                                static_cast<double>(m) - 1.0
                                - r2_over_sigma2[base_idx]
                            ) / static_cast<double>(n);
                            const double prev1 = lfunc[idx4(n - 1, m, j, i, norder, mord, ny, nx)];
                            const double prev2 = lfunc[idx4(n - 2, m, j, i, norder, mord, ny, nx)];
                            lfunc[idx4(n, m, j, i, norder, mord, ny, nx)] = (
                                (a_coeff + r_term) * prev1
                                - b_coeff * prev2
                            );
                        }
                    }
                }
            }

            std::vector<std::complex<double>> chi(total_size);
            for (int nn = 0; nn <= norder; ++nn) {
                for (int mm = nn; mm >= 0; mm -= 2) {
                    const int abs_m = std::abs(mm);
                    const int c1 = (nn - abs_m) / 2;
                    const int d1 = (nn + abs_m) / 2;
                    const double factorial_c1 = std::tgamma(static_cast<double>(c1) + 1.0);
                    const double factorial_d1 = std::tgamma(static_cast<double>(d1) + 1.0);
                    double coeff_ratio = 0.0;
                    if (factorial_d1 != 0.0) {
                        coeff_ratio = factorial_c1 / factorial_d1;
                    }
                    const double sign = (d1 % 2 == 0) ? 1.0 : -1.0;
                    const double sqrt_coeff = std::sqrt(coeff_ratio);
                    const std::complex<double> i_pow = pow_i(nn);
                    for (int j = 0; j < ny; ++j) {
                        for (int i = 0; i < nx; ++i) {
                            const size_t base_idx = static_cast<size_t>(j) * nx + static_cast<size_t>(i);
                            const double r_pow = std::pow(
                                r2_over_sigma2[base_idx],
                                static_cast<double>(abs_m) / 2.0
                            );
                            const double laguerre = lfunc[idx4(
                                c1, abs_m, j, i, norder, mord, ny, nx
                            )];
                            std::complex<double> eul_pow = 1.0;
                            if (abs_m > 0) {
                                eul_pow = std::pow(eulfunc[base_idx], abs_m);
                            }
                            const std::complex<double> value = (
                                sign
                                * sqrt_coeff
                                * laguerre
                                * r_pow
                                * gauss[base_idx]
                                * i_pow
                            );
                            chi[idx4(nn, mm, j, i, norder, mord, ny, nx)] = value * eul_pow;
                        }
                    }
                }
            }

            // Divide Gaussian kernel by npix**2 for inverse FFT normalization
            const double norm = static_cast<double>(npix) * static_cast<double>(npix);
            for (auto& val : chi) {
                val /= norm;
            }

            return {ny, nx, norder, std::move(chi)};
        }
    } // namespace fpfs_detail

    inline py::object
    gauss_kernel_rfft(
        int ny,
        int nx,
        double sigma,
        double kmax,
        bool return_grid = false
    ) {
        const int nx_r = nx / 2 + 1;
        py::array_t<double> kernel({ny, nx_r});
        py::array_t<double> ygrid({ny, nx_r});
        py::array_t<double> xgrid({ny, nx_r});

        auto k = kernel.mutable_unchecked<2>();
        auto yg = ygrid.mutable_unchecked<2>();
        auto xg = xgrid.mutable_unchecked<2>();

        std::vector<double> xfreq = fpfs_detail::make_rfftfreq(nx);
        std::vector<double> yfreq = fpfs_detail::make_fftfreq(ny);

        const double kmax2 = kmax * kmax;
        const double sigma2 = sigma * sigma;

        for (int j = 0; j < ny; ++j) {
            const double ky = yfreq[j];
            for (int i = 0; i < nx_r; ++i) {
                const double kx = xfreq[i];
                const double r2 = kx * kx + ky * ky;
                const double mask = (r2 <= kmax2) ? 1.0 : 0.0;
                k(j, i) = std::exp(-r2 / (2.0 * sigma2)) * mask;
                yg(j, i) = ky;
                xg(j, i) = kx;
            }
        }

        if (!return_grid) {
            return kernel;
        }

        return py::make_tuple(kernel, py::make_tuple(ygrid, xgrid));
    }

    inline double
    m00_to_flux(
        double m00,
        double sigma_arcsec,
        double pixel_scale
    ) {
        const double sigma_use = sigma_arcsec / std::sqrt(2.0);
        const double sigma_pix = sigma_use / pixel_scale;
        const double ff = 4.0 * M_PI * sigma_pix * sigma_pix;
        return m00 * pixel_scale * pixel_scale * ff;
    }

    inline py::array_t<double>
    m00_to_flux(
        const py::array_t<double>& m00,
        double sigma_arcsec,
        double pixel_scale
    ) {
        const double factor = m00_to_flux(
            1.0,
            sigma_arcsec,
            pixel_scale
        );

        int nn = m00.shape(0);
        auto in_r = m00.unchecked<1>();
        py::array_t<double> flux(nn);
        auto out_r = flux.mutable_unchecked<1>();
        for (py::ssize_t i = 0; i < nn; ++i) {
            out_r(i) = in_r(i) * factor;
        }
        return flux;
    }

    inline py::array_t<std::complex<double>>
    shapelets2d_func(int npix, int norder, double sigma, double kmax) {
        fpfs_detail::ShapeletResult result = fpfs_detail::compute_shapelet_modes(
            npix, norder, sigma, kmax
        );
        const int ny = result.ny;
        const int nx = result.nx;
        const int nmode = (norder + 1) * (norder + 1);
        py::array_t<std::complex<double>> chi({nmode, ny, nx});
        auto out = chi.mutable_unchecked<3>();
        for (int nn = 0; nn <= norder; ++nn) {
            for (int mm = 0; mm <= norder; ++mm) {
                const int mode_index = nn * (norder + 1) + mm;
                for (int j = 0; j < ny; ++j) {
                    for (int i = 0; i < nx; ++i) {
                        const size_t idx = fpfs_detail::idx4(
                            nn, mm, j, i, norder, norder, ny, nx
                        );
                        out(mode_index, j, i) = result.chi[idx];
                    }
                }
            }
        }
        return chi;
    }

    inline py::array_t<double>
    shapelets2d(int norder, int npix, double sigma, double kmax) {
        fpfs_detail::ShapeletResult result = fpfs_detail::compute_shapelet_modes(
            npix, norder, sigma, kmax
        );
        const int ny = result.ny;
        const int nx = result.nx;
        static constexpr std::array<int, 12> indices = {
            0, 14, 16, 16, 28, 30, 30, 32, 32, 42, 46, 46
        };
        static constexpr std::array<bool, 12> use_imag = {
            false, false, false, true, false, false, true,
            false, true, false, false, true
        };

        py::array_t<double> chi({12, ny, nx});
        auto out = chi.mutable_unchecked<3>();
        for (size_t mode = 0; mode < indices.size(); ++mode) {
            const int idx_mode = indices[mode];
            const int nn = idx_mode / (norder + 1);
            const int mm = idx_mode % (norder + 1);
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const size_t idx = fpfs_detail::idx4(
                        nn, mm, j, i, norder, norder, ny, nx
                    );
                    const std::complex<double>& value = result.chi[idx];
                    out(mode, j, i) = use_imag[mode] ? value.imag() : value.real();
                }
            }
        }
        return chi;
    }

    inline py::array_t<std::complex<double>>
    detlets2d(int npix, double sigma, double kmax) {
        const int ny = npix;
        const int nx = npix / 2 + 1;
        const int det_nrot = 4;
        const int nmode = 3 * det_nrot;

        py::array_t<std::complex<double>> psi({nmode, ny, nx});
        auto out = psi.mutable_unchecked<3>();

        std::vector<double> xfreq = fpfs_detail::make_rfftfreq(npix);
        std::vector<double> yfreq = fpfs_detail::make_fftfreq(npix);

        std::vector<double> gauss(static_cast<size_t>(ny) * nx, 0.0);
        std::vector<double> k1grid(static_cast<size_t>(ny) * nx, 0.0);
        std::vector<double> k2grid(static_cast<size_t>(ny) * nx, 0.0);

        const double kmax2 = kmax * kmax;
        const double sigma2 = sigma * sigma;

        for (int j = 0; j < ny; ++j) {
            const double ky = yfreq[j];
            for (int i = 0; i < nx; ++i) {
                const double kx = xfreq[i];
                const double r2 = kx * kx + ky * ky;
                const size_t idx = static_cast<size_t>(j) * nx + static_cast<size_t>(i);
                if (r2 <= kmax2) {
                    gauss[idx] = std::exp(-r2 / (2.0 * sigma2));
                } else {
                    gauss[idx] = 0.0;
                }
                k1grid[idx] = kx;
                k2grid[idx] = ky;
            }
        }

        const double norm = static_cast<double>(npix) * static_cast<double>(npix);
        std::vector<double> q1(static_cast<size_t>(ny) * nx, 0.0);
        std::vector<double> q2(static_cast<size_t>(ny) * nx, 0.0);
        std::vector<std::complex<double>> d1(static_cast<size_t>(ny) * nx);
        std::vector<std::complex<double>> d2(static_cast<size_t>(ny) * nx);

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const size_t idx = static_cast<size_t>(j) * nx + static_cast<size_t>(i);
                const double gaussian = gauss[idx] / norm;
                const double k1 = k1grid[idx];
                const double k2 = k2grid[idx];
                q1[idx] = ((k1 * k1) - (k2 * k2)) / sigma2 * gaussian;
                q2[idx] = (2.0 * k1 * k2) / sigma2 * gaussian;
                d1[idx] = std::complex<double>(0.0, -k1) * gaussian;
                d2[idx] = std::complex<double>(0.0, -k2) * gaussian;
                gauss[idx] = gaussian;
            }
        }

        for (int irot = 0; irot < det_nrot; ++irot) {
            const double angle = 2.0 * M_PI * static_cast<double>(irot) / det_nrot;
            const double cx = std::cos(angle);
            const double cy = std::sin(angle);
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const size_t idx = static_cast<size_t>(j) * nx + static_cast<size_t>(i);
                    const double k1 = k1grid[idx];
                    const double k2 = k2grid[idx];
                    const double phase = k1 * cx + k2 * cy;
                    const std::complex<double> foub = std::polar(1.0, phase);

                    const std::complex<double> psi0 = gauss[idx] * (1.0 - foub);
                    const std::complex<double> term1 = q1[idx] + cx * d1[idx] - cy * d2[idx];
                    const std::complex<double> term2 = q2[idx] + cy * d1[idx] + cx * d2[idx];

                    out(0 * det_nrot + irot, j, i) = psi0;
                    out(1 * det_nrot + irot, j, i) = q1[idx] - term1 * foub;
                    out(2 * det_nrot + irot, j, i) = q2[idx] - term2 * foub;
                }
            }
        }

        return psi;
    }

    inline double
    get_kmax(
        const py::array_t<double>& psf_pow,
        double sigma,
        double kmax_thres = 1e-20
    ) {
        auto psf = psf_pow.unchecked<2>();
        const int npix = static_cast<int>(psf.shape(0));
        const int nx = static_cast<int>(psf.shape(1));

        int nx_full = (nx - 1) * 2;
        if (nx_full <= 0) {
            nx_full = nx;
        }
        std::vector<double> xfreq = fpfs_detail::make_rfftfreq(nx_full);
        std::vector<double> yfreq = fpfs_detail::make_fftfreq(npix);

        const double sigma2 = sigma * sigma;
        const double kmax = M_PI;
        const double kmax2 = kmax * kmax;

        double min_r = std::numeric_limits<double>::max();
        bool found = false;

        for (int j = 0; j < npix; ++j) {
            const double ky = yfreq[j];
            for (int i = 0; i < nx; ++i) {
                const double kx = xfreq[i];
                const double r2 = kx * kx + ky * ky;
                if (r2 > kmax2) {
                    continue;
                }
                const double gaussian = std::exp(-r2 / (2.0 * sigma2));
                double psf_val = psf(j, i);
                double ratio;
                if (psf_val == 0.0) {
                    ratio = std::numeric_limits<double>::infinity();
                } else {
                    ratio = gaussian / psf_val;
                }
                if (ratio < kmax_thres) {
                    const double r = std::sqrt(r2);
                    if (r < min_r) {
                        min_r = r;
                        found = true;
                    }
                }
            }
        }

        if (!found) {
            return static_cast<double>(npix / 2 - 1);
        }

        const double dk = 2.0 * M_PI / static_cast<double>(npix);
        double kmax_pix = std::round(min_r / dk);
        const double min_pix = static_cast<double>(npix / 5);
        const double max_pix = static_cast<double>(npix / 2 - 1);
        if (kmax_pix < min_pix) {
            kmax_pix = min_pix;
        }
        if (kmax_pix > max_pix) {
            kmax_pix = max_pix;
        }
        return kmax_pix;
    }

    void pyExportFpfsBase(py::module_& fpfs);
}

#endif // ANACAL_FPFS_BASE_H
