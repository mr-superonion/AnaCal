#ifndef ANACAL_FPFS_FORCE_H
#define ANACAL_FPFS_FORCE_H

#include <algorithm>
#include <cstring>

#include "../psfmodel.h"
#include "../renoise.h"
#include "catalog.h"

namespace anacal {
    namespace fpfs {

    // Lower bound on the per-kernel flux uncertainty; MUST match
    // FLUX_ERR_MIN in python/anacal/fpfs.py (a noiseless image gives
    // std_m00 == 0, which would make s2n and its shear response infinite).
    inline constexpr double flux_err_min = 1e-8;

    // One finished catalog row for a single shapelet kernel: the
    // FpfsShape fields, then the columns the Python assembly used to
    // append (m00_err and the shapelet flux family: flux, dflux_dg,
    // flux_err, s2n, ds2n_dg), in the same order.  Kept a flat
    // all-double standard-layout struct so the FpfsShape prefix can be
    // filled with one memcpy and the numpy dtype is a plain packed f8
    // record.  The layout is the same for both shape orders (see
    // ForceTask): with the 9-mode basis the 4th-order estimator q and
    // its responses (q1, q2, dq*_dg*, dm42*_dg*) are NaN.
    struct FpfsCatRow {
        double e1;
        double de1_dg1;
        double de1_dg2;
        double e2;
        double de2_dg1;
        double de2_dg2;
        double q1;
        double dq1_dg1;
        double dq1_dg2;
        double q2;
        double dq2_dg1;
        double dq2_dg2;
        double m00;
        double dm00_dg1;
        double dm00_dg2;
        double m20;
        double dm20_dg1;
        double dm20_dg2;
        double m22c;
        double dm22c_dg1;
        double dm22c_dg2;
        double m22s;
        double dm22s_dg1;
        double dm22s_dg2;
        double m42c;
        double dm42c_dg1;
        double dm42c_dg2;
        double m42s;
        double dm42s_dg1;
        double dm42s_dg2;
        double m00_err;
        double flux;
        double dflux_dg1;
        double dflux_dg2;
        double flux_err;
        double s2n;
        double ds2n_dg1;
        double ds2n_dg2;
    };
    static_assert(
        std::is_standard_layout<FpfsCatRow>::value &&
        std::is_standard_layout<FpfsShape>::value,
        "memcpy prefix fill requires standard layout"
    );
    static_assert(
        offsetof(FpfsCatRow, m00_err) == sizeof(FpfsShape),
        "FpfsCatRow must start with the FpfsShape fields"
    );

    // Deconvolve the shapelet filter stack by a PSF spectrum held in a
    // std::vector, writing into a reusable scratch vector: the same
    // computation as deconvolve_filter in image.cpp, made allocation-free
    // so the ForceTask per-source-PSF loop can run it with the GIL
    // released.
    inline void
    deconvolve_filter_into(
        const std::complex<double>* f_ptr,
        const std::vector<std::complex<double>>& parr,
        double scale,
        double klim,
        int nky,
        int nkx,
        int ncol,
        std::vector<std::complex<double>>& out
    ) {
        const double dky = 2.0 * M_PI / nky / scale;
        const double dkx = 2.0 * M_PI / (2 * (nkx - 1)) / scale;
        const double p0 = klim * klim;
        const double v_test = parr[0].imag();
        if ((v_test < 0 ? -v_test : v_test) > 1e-10) {
            throw std::runtime_error(
                "Input PSF image is not real in configuration space"
            );
        }
        const double min_deconv_value = min_deconv_ratio * parr[0].real();
        out.resize(
            static_cast<std::size_t>(nky) * nkx * ncol
        );
        for (int j = 0; j < nky; ++j) {
            double ky = ((j < nky / 2) ? j : (j - nky)) * dky;
            for (int i = 0; i < nkx; ++i) {
                double kx = i * dkx;
                double r2 = kx * kx + ky * ky;
                const std::size_t base =
                    (static_cast<std::size_t>(j) * nkx + i) * ncol;
                if (r2 > p0) {
                    for (int icol = 0; icol < ncol; icol++) {
                        out[base + icol] = 0;
                    }
                } else {
                    std::complex<double> val;
                    const std::complex<double>& pv = parr[
                        static_cast<std::size_t>(j) * nkx + i
                    ];
                    double abs_kval = std::abs(pv);
                    if (abs_kval < min_deconv_value) {
                        val = 1.0 / min_deconv_value;
                    } else {
                        val = 1.0 / pv;
                    }
                    for (int icol = 0; icol < ncol; icol++) {
                        out[base + icol] = f_ptr[base + icol] * val;
                    }
                }
            }
        }
        return;
    };

    // Forced FPFS measurement for one shapelet kernel, structured like
    // task::Task: configuration in the constructor, then process_image
    // takes the image plus detected positions and returns the finished
    // catalog, with the whole per-source loop running with the GIL
    // released (the covariance scalar std_m00 stays a Python-side input:
    // it is one number per (PSF, kernel, variance) and its numpy
    // pairwise-summed integrals would not be bit-identical if ported).
    //
    // ``psf_array`` is a single (ny, nx) stamp used for every source
    // (the per-cell PSF).  A spatially varying PSF comes exclusively
    // through ``psf_model`` (psfmodel::PerSourcePsf -- coadd or stamp
    // grid), drawn inside the GIL-released loop: pre-drawn Python
    // stamp stacks are not supported.
    //
    // The output columns are named by the CALLER: ``out_dtype`` is a
    // packed all-float64 numpy dtype whose fields are the final column
    // names (kernel token and band prefix included), so no rename ever
    // happens downstream.  Field order is fixed by FpfsCatRow.
    //
    // The shape order is set by the filter stack: 9 modes (through
    // M44: m00, m20, m22c/s, m40, m42c/s, m44c/s) give the 2nd-order
    // estimator e = m22 / (m00 + c0) with its full response and the
    // trace-size response dm20/dg = -sqrt(6) m42, while q and its
    // responses are NaN; 12 modes (plus m60, m64c/s) add the 4th-order
    // estimator q = m42 / (m00 + c0) and its response.  The 9 measured
    // modes are identical between the two.
    // Option-2 (spectral renoising) noise filters from the deconvolved
    // data filter fg = chi / p (nky x nkx x ncol, contiguous): 3 * ncol
    // columns [eta0, tau1, tau2] * fg applied to the transform of the pure-noise
    // cutout eps', plus the filler filter rho * fg applied to the
    // transform of the second realization eps''.  The kernels are those
    // of the CELL (symmetrized once per call with the per-cell PSF,
    // shared with the detection stage); fg alone follows the source
    // PSF.  They are half-plane (n, n / 2 + 1) arrays on the cutout
    // grid, so their layout is that of fg.
    inline void
    spectral_noise_filters(
        const std::complex<double>* fg,
        int nky,
        int nkx,
        int ncol,
        const renoise::RenoiseKernels& rk,
        std::vector<std::complex<double>>& add,
        // the filler filter rho * fg, applied to the transform of the
        // SECOND noise realization eps'' (the filler is applied in
        // Fourier space; the image itself carries no filler)
        std::vector<std::complex<double>>& addrho
    ) {
        if ((rk.n != nky) || (renoise::half_width(rk.n) != nkx)) {
            throw std::runtime_error(
                "FPFS Error: renoise kernels are on a " +
                std::to_string(rk.n) + " grid but the stamp is " +
                std::to_string(nky)
            );
        }
        const std::size_t nf = static_cast<std::size_t>(nky) * nkx;
        const std::size_t nc = static_cast<std::size_t>(ncol);
        add.resize(nf * 3 * nc);
        addrho.resize(nf * nc);
        for (std::size_t idx = 0; idx < nf; ++idx) {
            const std::complex<double>* f = fg + idx * nc;
            std::complex<double>* a = add.data() + idx * 3 * nc;
            std::complex<double>* ar = addrho.data() + idx * nc;
            for (std::size_t c = 0; c < nc; ++c) {
                a[c] = f[c] * rk.eta0[idx];
                a[nc + c] = f[c] * rk.tau1[idx];
                a[2 * nc + c] = f[c] * rk.tau2[idx];
                ar[c] = f[c] * rk.rho[idx];
            }
        }
        return;
    };

    // The assigned response: the moments of tau_j * noise enter the
    // shear derivatives of the modes directly.
    inline void
    add_assigned_response(
        FpfsShapeletsResponse& dg,
        const FpfsShapelets& k1,
        const FpfsShapelets& k2
    ) {
        dg.dm00_dg1 += k1.m00;
        dg.dm20_dg1 += k1.m20;
        dg.dm22c_dg1 += k1.m22c;
        dg.dm22s_dg1 += k1.m22s;
        dg.dm42c_dg1 += k1.m42c;
        dg.dm42s_dg1 += k1.m42s;
        dg.dm00_dg2 += k2.m00;
        dg.dm20_dg2 += k2.m20;
        dg.dm22c_dg2 += k2.m22c;
        dg.dm22s_dg2 += k2.m22s;
        dg.dm42c_dg2 += k2.m42c;
        dg.dm42s_dg2 += k2.m42s;
        return;
    };

    // One finished row of the ADAPTIVE kernel ("afpfs"): the FpfsCatRow
    // fields, then the per-source kernel size and its shear response.
    struct AfpfsCatRow {
        FpfsCatRow base;
        double sigma;
        double dsigma_dg1;
        double dsigma_dg2;
    };
    static_assert(
        std::is_standard_layout<AfpfsCatRow>::value &&
        offsetof(AfpfsCatRow, sigma) == sizeof(FpfsCatRow),
        "AfpfsCatRow must start with the FpfsCatRow fields"
    );

    // Per-source kernel sizes of the adaptive measurement, on a grid of
    // kernel sizes uniform in log sigma.  The shapelet basis is
    // tabulated at the nodes, ``table`` (n_nodes, nky, nkx, ncol); a
    // source is measured with the four nodes around its size and its
    // modes are the Catmull-Rom cubic through them (C1 in sigma, so the
    // size derivative has no jumps where a node is crossed).  That
    // derivative is the slope of the interpolant, so the shear response
    // through dsigma/dg is exactly the derivative of the estimator that
    // is evaluated.  Sizes must lie in [grid[1], grid[n_nodes - 2]].
    inline constexpr int adaptive_nst = 4;
    struct AdaptiveKernel {
        const std::complex<double>* table;
        int n_nodes;             // >= 4
        const double* grid;      // node sizes, increasing [arcsec]
        const double* std_grid;  // std_m00 at the nodes
        const double* sig;       // per-source kernel size [arcsec]
        const double* dsig1;     // and its shear response
        const double* dsig2;
    };

    // Catmull-Rom weights on [0, 1] for the nodes (k-1, k, k+1, k+2)
    // around the interval [k, k+1], and their t-derivatives.
    inline void
    catmull_rom(double t, double* w, double* dw) {
        const double t2 = t * t;
        const double t3 = t2 * t;
        w[0] = 0.5 * (-t + 2.0 * t2 - t3);
        w[1] = 0.5 * (2.0 - 5.0 * t2 + 3.0 * t3);
        w[2] = 0.5 * (t + 4.0 * t2 - 3.0 * t3);
        w[3] = 0.5 * (-t2 + t3);
        dw[0] = 0.5 * (-1.0 + 4.0 * t - 3.0 * t2);
        dw[1] = 0.5 * (-10.0 * t + 9.0 * t2);
        dw[2] = 0.5 * (1.0 + 8.0 * t - 9.0 * t2);
        dw[3] = 0.5 * (-2.0 * t + 3.0 * t2);
        return;
    };

    // Modes measured with one (interpolated) kernel and their derivative
    // with respect to the kernel size (zero for a fixed kernel).
    struct ModesDs {
        FpfsShapelets v;
        FpfsShapelets ds;
    };

    class ForceTask {
    private:
        ForceTask(const ForceTask&) = delete;
        ForceTask& operator=(const ForceTask&) = delete;
        double fft_ratio;
    public:
        int npix;
        double scale;
        double sigma_arcsec;
        double klim;
        double c0;

        ForceTask(
            int npix,
            double scale,
            double sigma_arcsec,
            double klim,
            double c0
        ) : npix(npix), scale(scale), sigma_arcsec(sigma_arcsec),
            klim(klim), c0(c0)
        {
            if ((sigma_arcsec <= 0) || (sigma_arcsec > 5.0)) {
                throw std::runtime_error(
                    "FPFS Error: invalid input sigma_arcsec"
                );
            }
            this->fft_ratio = 1.0 / scale / scale;
        };

        // Fixed kernel: ``filter_image`` is the (nky, nkx, ncol) basis
        // of the kernel ``sigma_arcsec``.
        py::array
        process_image(
            const py::array_t<pixel_t>& gal_array,
            const py::array_t<double>& psf_array,
            const py::array_t<std::complex<double>>& filter_image,
            const py::array_t<Position>& det,
            double std_m00,
            const py::object& out_dtype,
            const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
            // second, independent noise realization eps'' (spectral
            // renoising): the spin-2 filler rho * eps'' is applied in
            // Fourier space per stamp; the image carries no filler
            const std::optional<py::array_t<pixel_t>>& noise2_array=std::nullopt,
            // Per-source mask flags + threshold: sources with
            // n_mask_base > n_mask_base_max are SKIPPED -- their output
            // rows are zero-filled, keeping the catalog row-aligned
            // with ``det``.  The psf_invalid_mask_value sentinel (1.0,
            // "PSF invalid in some band") skips ALWAYS, threshold or
            // not: a source without a usable PSF anywhere is dropped
            // from every band, not only from the band that found out.
            const std::optional<
                py::array_t<float, py::array::c_style>
            >& n_mask_base=std::nullopt,
            const std::optional<double>& n_mask_base_max=std::nullopt,
            // Native per-source PSF: stamps are drawn INSIDE the
            // GIL-released loop by the model (no Python per-galaxy
            // drawing), and a source outside the model's coverage is
            // skipped with n_mask_base set to 1.0 in place (the array
            // must then be a writable int32 array).  ``det`` positions
            // are local image coordinates; the offsets map them into
            // the model's frame (the exposure bbox minimum).
            const std::shared_ptr<psfmodel::PerSourcePsf>& psf_model=nullptr,
            double psf_offset_x=0.0,
            double psf_offset_y=0.0,
            // Spectral renoising (Option 2): one NoiseSpectrum on the
            // npix grid plus (c, delta).  Null: the rotated-noise
            // correction (Option 1) whenever noise_array is given.
            const renoise::Setup* renoise=nullptr
        ) {
            if (filter_image.ndim() != 3) {
                throw std::runtime_error(
                    "FPFS Error: Input filter image must be 3-dimensional."
                );
            }
            return this->run(
                &filter_image, nullptr,
                gal_array, psf_array, det, std_m00, out_dtype,
                noise_array, noise2_array, n_mask_base, n_mask_base_max,
                psf_model, psf_offset_x, psf_offset_y, renoise
            );
        };

        // Adaptive kernel ("afpfs"): every source is measured with its
        // own kernel size ``sigma`` (with shear response dsigma_dg),
        // interpolated (Catmull-Rom cubic in log sigma) on the
        // tabulated basis ``filter_table`` (n_nodes, nky, nkx, ncol)
        // over the node sizes ``sigma_grid`` (uniform in log);
        // ``std_m00_grid`` is the mode-noise scalar at the nodes.  The
        // output rows are AfpfsCatRow (the FpfsCatRow fields, then
        // sigma, dsigma_dg1, dsigma_dg2); see adaptive_catalog_columns.
        // Sizes outside [grid[1], grid[n_nodes - 2]] are measured at
        // that bound (the Python side keeps them inside with smooth
        // clamps).
        py::array
        process_image_adaptive(
            const py::array_t<pixel_t>& gal_array,
            const py::array_t<double>& psf_array,
            const py::array_t<std::complex<double>, py::array::c_style>&
                filter_table,
            const py::array_t<double, py::array::c_style>& sigma_grid,
            const py::array_t<double, py::array::c_style>& std_m00_grid,
            const py::array_t<double, py::array::c_style>& sigma,
            const py::array_t<double, py::array::c_style>& dsigma_dg1,
            const py::array_t<double, py::array::c_style>& dsigma_dg2,
            const py::array_t<Position>& det,
            const py::object& out_dtype,
            const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
            const std::optional<py::array_t<pixel_t>>& noise2_array=std::nullopt,
            const std::optional<
                py::array_t<float, py::array::c_style>
            >& n_mask_base=std::nullopt,
            const std::optional<double>& n_mask_base_max=std::nullopt,
            const std::shared_ptr<psfmodel::PerSourcePsf>& psf_model=nullptr,
            double psf_offset_x=0.0,
            double psf_offset_y=0.0,
            const renoise::Setup* renoise=nullptr
        ) {
            if (filter_table.ndim() != 4) {
                throw std::runtime_error(
                    "FPFS Error: filter_table must be "
                    "(n_nodes, nky, nkx, ncol)."
                );
            }
            const int n_nodes = static_cast<int>(filter_table.shape(0));
            if (n_nodes < adaptive_nst) {
                throw std::runtime_error(
                    "FPFS Error: filter_table needs at least four nodes."
                );
            }
            if ((sigma_grid.ndim() != 1) ||
                (sigma_grid.shape(0) != n_nodes) ||
                (std_m00_grid.ndim() != 1) ||
                (std_m00_grid.shape(0) != n_nodes)) {
                throw std::runtime_error(
                    "FPFS Error: sigma_grid and std_m00_grid must hold "
                    "one value per node."
                );
            }
            const ssize_t nrow = det.shape(0);
            if ((sigma.ndim() != 1) || (sigma.shape(0) != nrow) ||
                (dsigma_dg1.ndim() != 1) || (dsigma_dg1.shape(0) != nrow) ||
                (dsigma_dg2.ndim() != 1) || (dsigma_dg2.shape(0) != nrow)) {
                throw std::runtime_error(
                    "FPFS Error: sigma, dsigma_dg1, dsigma_dg2 must hold "
                    "one value per source."
                );
            }
            const double* grid = sigma_grid.data();
            for (int i = 0; i + 1 < n_nodes; ++i) {
                if (!(grid[i + 1] > grid[i])) {
                    throw std::runtime_error(
                        "FPFS Error: sigma_grid must increase strictly."
                    );
                }
            }
            if ((grid[0] <= 0) || (grid[n_nodes - 1] > 5.0)) {
                throw std::runtime_error(
                    "FPFS Error: sigma_grid must lie in (0, 5] arcsec."
                );
            }
            AdaptiveKernel ad{
                filter_table.data(), n_nodes, grid, std_m00_grid.data(),
                sigma.data(), dsigma_dg1.data(), dsigma_dg2.data()
            };
            return this->run(
                &filter_table, &ad,
                gal_array, psf_array, det, 0.0, out_dtype,
                noise_array, noise2_array, n_mask_base, n_mask_base_max,
                psf_model, psf_offset_x, psf_offset_y, renoise
            );
        };

    private:
        // The measurement proper, shared by the fixed and the adaptive
        // kernel.  ``filt`` is the basis array: (nky, nkx, ncol) for a
        // fixed kernel, (n_nodes, nky, nkx, ncol) with ``ad`` set.
        py::array
        run(
            const py::array* filt,
            const AdaptiveKernel* ad,
            const py::array_t<pixel_t>& gal_array,
            const py::array_t<double>& psf_array,
            const py::array_t<Position>& det,
            double std_m00,
            const py::object& out_dtype,
            const std::optional<py::array_t<pixel_t>>& noise_array,
            const std::optional<py::array_t<pixel_t>>& noise2_array,
            const std::optional<
                py::array_t<float, py::array::c_style>
            >& n_mask_base,
            const std::optional<double>& n_mask_base_max,
            const std::shared_ptr<psfmodel::PerSourcePsf>& psf_model,
            double psf_offset_x,
            double psf_offset_y,
            const renoise::Setup* renoise
        ) {
            const bool adaptive = (ad != nullptr);
            // nst stacks of ncol modes per source: the adaptive kernel
            // measures the four nodes around the source's size
            const int nst = adaptive ? adaptive_nst : 1;
            const int d0 = adaptive ? 1 : 0;
            const int nky = static_cast<int>(filt->shape(d0));
            const int nkx = static_cast<int>(filt->shape(d0 + 1));
            const int ncol = static_cast<int>(filt->shape(d0 + 2));
            if ((ncol != 9) && (ncol != 12)) {
                throw std::runtime_error(
                    "FPFS Error: expected the 9-mode (through m44) or "
                    "12-mode (through m64) shapelet basis."
                );
            }
            // 4th-order estimator available only with m60, m64c, m64s
            const bool fourth_order = (ncol == 12);
            constexpr double qnan = std::numeric_limits<double>::quiet_NaN();
            const std::size_t nf = static_cast<std::size_t>(nky) * nkx;
            // one stack of ncol filter columns
            const std::size_t stack = nf * static_cast<std::size_t>(ncol);
            // Per-source PSFs come exclusively from ``psf_model``
            // (drawn natively inside the loop below); everything else
            // shares the single 2-D ``psf_array`` stamp.
            const bool native_psf = (psf_model != nullptr);
            if (det.ndim() != 1) {
                throw std::runtime_error(
                    "FPFS Error: det must be a 1-D (y, x) record array."
                );
            }
            const ssize_t nrow = det.shape(0);
            if (psf_array.ndim() != 2) {
                // Pre-drawn per-source stamp stacks are NOT supported:
                // per-source PSFs come exclusively from ``psf_model``
                // (drawn natively inside the loop); otherwise a single
                // 2-D per-cell stamp is used for every source.
                throw std::runtime_error(
                    "FPFS Error: psf_array must be a single 2-D stamp; "
                    "per-source PSFs are provided via psf_model."
                );
            }
            const bool do_mask_cut = (
                n_mask_base.has_value() && n_mask_base_max.has_value()
            );
            if (n_mask_base.has_value() && n_mask_base->shape(0) != nrow) {
                throw std::runtime_error(
                    "FPFS Error: n_mask_base must hold one value per source."
                );
            }
            if ((native_psf || adaptive) &&
                !(filt->flags() & py::array::c_style)) {
                throw std::runtime_error(
                    "FPFS Error: filter image must be C-contiguous."
                );
            }

            // The caller supplies the FINAL column names; only the packed
            // all-f8 layout of FpfsCatRow / AfpfsCatRow is fixed here.
            const py::dtype dt = py::dtype::from_args(out_dtype);
            const ssize_t row_size = adaptive
                ? static_cast<ssize_t>(sizeof(AfpfsCatRow))
                : static_cast<ssize_t>(sizeof(FpfsCatRow));
            if (dt.itemsize() != row_size) {
                throw std::runtime_error(
                    "FPFS Error: out_dtype does not match the catalog row."
                );
            }
            py::array out(dt, nrow);
            char* rows = static_cast<char*>(out.mutable_data());

            ImageLease lease(npix, npix, scale, true);
            Image& img_obj = lease.get();
            const bool has_noise = noise_array.has_value();
            const bool spectral = (renoise != nullptr);
            if (spectral) {
                if (!has_noise) {
                    throw std::runtime_error(
                        "FPFS Error: spectral renoising needs a noise image"
                    );
                }
                if (!noise2_array.has_value()) {
                    throw std::runtime_error(
                        "FPFS Error: spectral renoising needs a second "
                        "noise image (the filler realization eps'')"
                    );
                }
                if (renoise->spectra.size() != 1) {
                    throw std::runtime_error(
                        "FPFS Error: the renoise setup must hold exactly one "
                        "noise spectrum (the band being measured)"
                    );
                }
                if (renoise->spectra[0].n != npix) {
                    throw std::runtime_error(
                        "FPFS Error: the noise spectrum is on a " +
                        std::to_string(renoise->spectra[0].n) +
                        " grid but the stamp is " + std::to_string(npix)
                    );
                }
                if (renoise->spectra[0].symmetrized) {
                    throw std::runtime_error(
                        "FPFS Error: the renoise setup needs the raw "
                        "noise spectrum (make_noise_spectrum); the "
                        "symmetrization is per cell, here"
                    );
                }
            }
            // Option-2 combined filters [eta0, tau1, tau2] * fg and
            // rho * fg.  With a constant PSF and a fixed kernel they are
            // formed once, before the loop; with a native per-source PSF
            // model or the adaptive kernel only this recombination with
            // the per-source fg = chi / p runs inside the loop -- a
            // multiplication pass, no kernel construction (the kernels
            // are rk_cell, fixed per call).
            std::vector<std::complex<double>> fimg_add[adaptive_nst];
            std::vector<std::complex<double>> fimg_rho[adaptive_nst];

            // Spectral kernels of the CELL, built ONCE per call: the raw
            // spectrum is symmetrized with the per-cell ``psf_array`` --
            // the same PSF the detection stage used -- for the
            // constant-PSF and the per-source-PSF paths alike.  The
            // renoising fields rho * eps'' and eta_0 * eps' are part of
            // the image (one realization, the cell's), so their kernels
            // must not follow the source PSF; only the deconvolution
            // filter fg = chi / p is per source, exactly as for the
            // data pixels.
            renoise::RenoiseKernels rk_cell;
            if (spectral) {
                rk_cell = renoise::make_renoise_kernels(
                    renoise->spectra[0], psf_array, klim,
                    renoise->c, renoise->delta, true,
                    renoise->noise_rotated
                );
            }

            // Constant-PSF filters: data filter from the PSF, noise
            // filter from the PSF rotated 90 degrees in Fourier space.
            // ONE transform serves both -- draw_f() has already copied
            // the result out, so rotate90_f() can turn the buffer into
            // the rotated version in place.  (For a per-source PSF the
            // filters are rebuilt inside the loop instead.)  With the
            // adaptive kernel every interval's pair of stacks is
            // deconvolved once, into fg_table / fn_table.
            py::array_t<std::complex<double>> fimg_g;
            py::array_t<std::complex<double>> fimg_n;
            std::vector<std::complex<double>> fg_table;
            std::vector<std::complex<double>> fn_table;
            const std::complex<double>* filt_ptr =
                static_cast<const std::complex<double>*>(filt->data());
            if (!native_psf && !adaptive) {
                const py::array_t<std::complex<double>> filter_image =
                    py::reinterpret_borrow<
                        py::array_t<std::complex<double>>
                    >(*filt);
                img_obj.set_r(psf_array, false);
                img_obj.fft();
                const py::array_t<std::complex<double>> parr =
                    img_obj.draw_f();
                fimg_g = deconvolve_filter(
                    filter_image, parr, scale, klim
                );
                if (has_noise && !spectral) {
                    img_obj.rotate90_f();
                    const py::array_t<std::complex<double>> parr_n =
                        img_obj.draw_f();
                    fimg_n = deconvolve_filter(
                        filter_image, parr_n, scale, klim
                    );
                }
                if (spectral) {
                    // the filler filter rho * fg acts on the eps'' stamp
                    spectral_noise_filters(
                        fimg_g.data(), nky, nkx, ncol, rk_cell,
                        fimg_add[0], fimg_rho[0]
                    );
                }
            } else if (!native_psf && adaptive) {
                // every node's stack deconvolved once per call
                img_obj.set_r(psf_array, false);
                img_obj.fft();
                const std::vector<std::complex<double>> parr =
                    img_obj.draw_f_vec();
                std::vector<std::complex<double>> parr_n;
                if (has_noise && !spectral) {
                    img_obj.rotate90_f();
                    parr_n = img_obj.draw_f_vec();
                }
                std::vector<std::complex<double>> scratch;
                const std::size_t nn = static_cast<std::size_t>(ad->n_nodes);
                fg_table.resize(nn * stack);
                if (has_noise && !spectral) {
                    fn_table.resize(nn * stack);
                }
                for (std::size_t k = 0; k < nn; ++k) {
                    const std::complex<double>* src = filt_ptr + k * stack;
                    deconvolve_filter_into(
                        src, parr, scale, klim, nky, nkx, ncol, scratch
                    );
                    std::copy(
                        scratch.begin(), scratch.end(),
                        fg_table.begin() + k * stack
                    );
                    if (has_noise && !spectral) {
                        deconvolve_filter_into(
                            src, parr_n, scale, klim, nky, nkx, ncol, scratch
                        );
                        std::copy(
                            scratch.begin(), scratch.end(),
                            fn_table.begin() + k * stack
                        );
                    }
                }
            }
            const std::complex<double>* fg_ptr =
                (native_psf || adaptive) ? nullptr : fimg_g.data();
            const std::complex<double>* fn_ptr =
                (!native_psf && !adaptive && has_noise) ? fimg_n.data()
                                                        : nullptr;

            auto det_r = det.unchecked<1>();

            // Same arithmetic as the Python assembly: flux and its
            // responses use the vectorized m00_to_flux (multiply by the
            // unit-flux factor), flux_err the scalar overload.
            const double factor0 = m00_to_flux(1.0, this->sigma_arcsec);
            const double flux_err0 = std::max(
                m00_to_flux(std_m00, this->sigma_arcsec), flux_err_min
            );

            // Everything above allocates numpy arrays and so needs the
            // GIL; from here on the loop only reads the inputs through
            // unchecked accessors / raw pointers and writes into the
            // output buffer (the per-source deconvolution scratch is
            // plain C++ heap).  The braced scope ends the release BEFORE
            // the return statement touches out's refcount.
            // The writable mask pointer is acquired while the GIL is
            // still held (handle copy + writeability check); the raw
            // pointer stays valid for the whole call.
            float* mv_mut = nullptr;
            if (n_mask_base.has_value() && native_psf) {
                py::array_t<float, py::array::c_style> mv_arr =
                    *n_mask_base;
                mv_mut = mv_arr.mutable_data();
            }
            {
                ScopedGilRelease release;
                const float* mv_ptr =
                    n_mask_base.has_value() ? n_mask_base->data() : nullptr;
                const float mvmax = do_mask_cut
                    ? static_cast<float>(*n_mask_base_max) : 0.0f;
                std::vector<std::complex<double>> scratch_g[adaptive_nst];
                std::vector<std::complex<double>> scratch_n[adaptive_nst];
                std::vector<double> psf_scratch;
                if (native_psf) {
                    psf_scratch.resize(
                        static_cast<std::size_t>(npix) * npix
                    );
                }
                for (ssize_t j = 0; j < nrow; ++j) {
                    char* row_ptr = rows + j * row_size;
                    // Skip conditions.  Zero-filled rows keep the
                    // catalog aligned with the input positions.
                    //
                    // 1. The sentinel, ALWAYS and independently of any
                    //    configured threshold.  "No valid PSF model
                    //    here" is a property of the SOURCE, not of one
                    //    band: a source flagged by any band is dropped
                    //    from EVERY band, whether or not this band's
                    //    own model could have measured it.
                    // 2. n_mask_base over the configured threshold.
                    //
                    // The sentinel is 1.0, the top of the n_mask_base
                    // range, so it collides with a source whose kernel
                    // footprint lies entirely inside the mask.  That
                    // collision is harmless BY DESIGN: both readings of
                    // 1.0 mean "unusable", and both skip.
                    if (mv_ptr != nullptr &&
                        (mv_ptr[j] >= psfmodel::psf_invalid_mask_value ||
                         (do_mask_cut && mv_ptr[j] > mvmax))) {
                        std::memset(row_ptr, 0, row_size);
                        continue;
                    }
                    // The adaptive kernel: the interval [k0, k0 + 1]
                    // holding this source's size, its position t in
                    // log sigma, and d(t)/d(sigma) for the size
                    // derivative of the interpolant.
                    int k0 = 1;
                    double t = 0.0, dscale = 0.0;
                    double sig_j = this->sigma_arcsec;
                    double dsig1 = 0.0, dsig2 = 0.0;
                    if (adaptive) {
                        sig_j = ad->sig[j];
                        dsig1 = ad->dsig1[j];
                        dsig2 = ad->dsig2[j];
                        const double* g = ad->grid;
                        const int nn = ad->n_nodes;
                        const double lo = g[1];
                        const double hi = g[nn - 2];
                        double s_use = sig_j;
                        if (!(s_use > lo)) {
                            s_use = lo;
                        } else if (!(s_use < hi)) {
                            s_use = hi;
                        }
                        k0 = static_cast<int>(
                            std::upper_bound(g + 1, g + nn - 1, s_use) - g
                        ) - 1;
                        if (k0 > nn - 3) {
                            k0 = nn - 3;
                        }
                        const double h = std::log(g[k0 + 1] / g[k0]);
                        t = std::log(s_use / g[k0]) / h;
                        dscale = 1.0 / (h * s_use);
                    }
                    // the nst filter stacks of this source (nodes
                    // k0 - 1 .. k0 + 2 for the adaptive kernel)
                    const std::complex<double>* fg[adaptive_nst] = {
                        fg_ptr, nullptr, nullptr, nullptr
                    };
                    const std::complex<double>* fn[adaptive_nst] = {
                        fn_ptr, nullptr, nullptr, nullptr
                    };
                    const std::complex<double>* fsrc[adaptive_nst] = {
                        filt_ptr, nullptr, nullptr, nullptr
                    };
                    if (adaptive) {
                        for (int k = 0; k < nst; ++k) {
                            const std::size_t node =
                                static_cast<std::size_t>(k0 - 1 + k);
                            fsrc[k] = filt_ptr + node * stack;
                            if (!native_psf) {
                                fg[k] = fg_table.data() + node * stack;
                                if (has_noise && !spectral) {
                                    fn[k] = fn_table.data() + node * stack;
                                }
                            }
                        }
                    }
                    if (native_psf) {
                        // Draw the per-source PSF inside the released
                        // loop -- no Python per-galaxy drawing.  A
                        // source outside the model's coverage is the
                        // DM InvalidPsfError case: flag it 1.0 and
                        // skip.
                        const double mx = det_r(j).x + psf_offset_x;
                        const double my = det_r(j).y + psf_offset_y;
                        // contains() covers the "no input images here"
                        // case; the draw itself can still fail on a
                        // degenerate WCS linearization or an all-zero
                        // warp.  Both mean "no usable PSF for THIS
                        // source", so they are handled identically --
                        // and locally, because letting the exception
                        // escape would discard the whole cell's catalog
                        // over one pathological position.
                        bool psf_ok = psf_model->contains(mx, my);
                        if (psf_ok) {
                            try {
                                const psfmodel::Stamp stamp =
                                    psf_model->compute_image(mx, my);
                                psfmodel::resize_stamp_to(
                                    stamp, npix, psf_scratch.data()
                                );
                            } catch (const std::exception&) {
                                psf_ok = false;
                            }
                        }
                        if (!psf_ok) {
                            if (mv_mut) {
                                mv_mut[j] =
                                    psfmodel::psf_invalid_mask_value;
                            }
                            std::memset(row_ptr, 0, row_size);
                            continue;
                        }
                        img_obj.set_r_raw(
                            psf_scratch.data(), npix, npix, false
                        );
                        img_obj.fft();
                        const std::vector<std::complex<double>> parr =
                            img_obj.draw_f_vec();
                        for (int k = 0; k < nst; ++k) {
                            deconvolve_filter_into(
                                fsrc[k], parr, scale, klim,
                                nky, nkx, ncol, scratch_g[k]
                            );
                            fg[k] = scratch_g[k].data();
                        }
                        if (has_noise && !spectral) {
                            // same transform as above, rotated in place
                            img_obj.rotate90_f();
                            const std::vector<std::complex<double>> parr_n =
                                img_obj.draw_f_vec();
                            for (int k = 0; k < nst; ++k) {
                                deconvolve_filter_into(
                                    fsrc[k], parr_n, scale, klim,
                                    nky, nkx, ncol, scratch_n[k]
                                );
                                fn[k] = scratch_n[k].data();
                            }
                        }
                    }
                    const std::complex<double>* fadd[adaptive_nst] = {
                        nullptr, nullptr, nullptr, nullptr
                    };
                    const std::complex<double>* frho[adaptive_nst] = {
                        nullptr, nullptr, nullptr, nullptr
                    };
                    if (spectral) {
                        if (native_psf || adaptive) {
                            // per-cell kernels, per-source deconvolution:
                            // the noise fields are one fixed realization
                            // shared with the detection stage, so the
                            // kernels stay those of the cell (rk_cell,
                            // built once above) and only the filter
                            // fg = chi / p follows this source's PSF
                            // (and, adaptive, its kernel size)
                            for (int k = 0; k < nst; ++k) {
                                spectral_noise_filters(
                                    fg[k], nky, nkx, ncol, rk_cell,
                                    fimg_add[k], fimg_rho[k]
                                );
                            }
                        }
                        for (int k = 0; k < nst; ++k) {
                            fadd[k] = fimg_add[k].data();
                            frho[k] = fimg_rho[k].data();
                        }
                    }
                    const ModesDs xg = this->measure_at(
                        img_obj, gal_array, fg, ncol, nst, t, dscale,
                        det_r(j).y, det_r(j).x
                    );
                    FpfsShapelets x = xg.v;
                    FpfsShapelets dxds = xg.ds;
                    FpfsShapeletsResponse dgr;
                    if (spectral) {
                        // filler: rho * eps'' in Fourier space -- it plays
                        // the role of image noise, so its modes join the
                        // image's for the value AND the analytic response
                        const ModesDs xphi = this->measure_at(
                            img_obj, *noise2_array, frho,
                            ncol, nst, t, dscale,
                            det_r(j).y, det_r(j).x
                        );
                        // value: image + filler + added noise; response:
                        // the analytic response of image + filler plus
                        // the assigned response of the noise
                        ModesDs xw, xk1, xk2;
                        this->measure_noise_spectral(
                            img_obj, *noise_array, fadd,
                            ncol, nst, t, dscale,
                            det_r(j).y, det_r(j).x, xw, xk1, xk2
                        );
                        const FpfsShapelets xs = xg.v + xphi.v;
                        x = xs + xw.v;
                        dxds = xg.ds + xphi.ds + xw.ds;
                        dgr = measure_shapelets_dg(xs);
                        add_assigned_response(dgr, xk1.v, xk2.v);
                    } else {
                        std::optional<FpfsShapelets> yn = std::nullopt;
                        if (has_noise) {
                            const ModesDs xn = this->measure_at(
                                img_obj, *noise_array, fn,
                                ncol, nst, t, dscale,
                                det_r(j).y, det_r(j).x
                            );
                            x = xg.v + xn.v;
                            dxds = xg.ds + xn.ds;
                            yn = xn.v;
                        }
                        dgr = measure_shapelets_dg(x, yn);
                    }
                    if (adaptive) {
                        // the kernel size moves with the shear: the
                        // measured value's derivative along sigma times
                        // dsigma/dg, for every mode that carries a
                        // response
                        dgr.dm00_dg1 += dxds.m00 * dsig1;
                        dgr.dm00_dg2 += dxds.m00 * dsig2;
                        dgr.dm20_dg1 += dxds.m20 * dsig1;
                        dgr.dm20_dg2 += dxds.m20 * dsig2;
                        dgr.dm22c_dg1 += dxds.m22c * dsig1;
                        dgr.dm22c_dg2 += dxds.m22c * dsig2;
                        dgr.dm22s_dg1 += dxds.m22s * dsig1;
                        dgr.dm22s_dg2 += dxds.m22s * dsig2;
                        dgr.dm42c_dg1 += dxds.m42c * dsig1;
                        dgr.dm42c_dg2 += dxds.m42c * dsig2;
                        dgr.dm42s_dg1 += dxds.m42s * dsig1;
                        dgr.dm42s_dg2 += dxds.m42s * dsig2;
                    }
                    if (!fourth_order) {
                        // dm42/dg needs m60 and m64, which the 9-mode
                        // basis does not measure (their slots are zero):
                        // the 4th-order response is undefined, not zero.
                        dgr.dm42c_dg1 = qnan;
                        dgr.dm42c_dg2 = qnan;
                        dgr.dm42s_dg1 = qnan;
                        dgr.dm42s_dg2 = qnan;
                    }
                    const FpfsShape sh = measure_fpfs_shape(
                        this->c0, x, dgr
                    );
                    FpfsCatRow& row = *reinterpret_cast<FpfsCatRow*>(row_ptr);
                    std::memcpy(&row, &sh, sizeof(FpfsShape));
                    if (!fourth_order) {
                        // dq*_dg* are already NaN through dgr; q itself
                        // is dropped too, as an estimator without a
                        // response.  m42c/s stay: they are measured and
                        // carry dm20/dg.
                        row.q1 = qnan;
                        row.q2 = qnan;
                    }
                    if (!adaptive) {
                        row.m00_err = std_m00;
                        row.flux = sh.m00 * factor0;
                        row.dflux_dg1 = sh.dm00_dg1 * factor0;
                        row.dflux_dg2 = sh.dm00_dg2 * factor0;
                        row.flux_err = flux_err0;
                        row.s2n = row.flux / flux_err0;
                        row.ds2n_dg1 = row.dflux_dg1 / flux_err0;
                        row.ds2n_dg2 = row.dflux_dg2 / flux_err0;
                    } else {
                        // the flux factor 2 pi sigma^2 and the mode
                        // noise follow the source's kernel size, so both
                        // carry a dsigma/dg term
                        const double factor = m00_to_flux(1.0, sig_j);
                        const double dfactor = 4.0 * M_PI * sig_j;
                        const double* sg = ad->std_grid + (k0 - 1);
                        double cw[adaptive_nst], cdw[adaptive_nst];
                        catmull_rom(t, cw, cdw);
                        double std_j = 0.0, dstd_ds = 0.0;
                        for (int k = 0; k < adaptive_nst; ++k) {
                            std_j += cw[k] * sg[k];
                            dstd_ds += cdw[k] * sg[k];
                        }
                        dstd_ds *= dscale;
                        const double ferr_raw = m00_to_flux(std_j, sig_j);
                        const bool floored = !(ferr_raw > flux_err_min);
                        const double ferr = floored ? flux_err_min : ferr_raw;
                        const double dferr_ds = floored
                            ? 0.0 : (dstd_ds * factor + std_j * dfactor);
                        row.m00_err = std_j;
                        row.flux = sh.m00 * factor;
                        row.dflux_dg1 =
                            sh.dm00_dg1 * factor + sh.m00 * dfactor * dsig1;
                        row.dflux_dg2 =
                            sh.dm00_dg2 * factor + sh.m00 * dfactor * dsig2;
                        row.flux_err = ferr;
                        row.s2n = row.flux / ferr;
                        row.ds2n_dg1 = row.dflux_dg1 / ferr
                            - row.flux * dferr_ds * dsig1 / (ferr * ferr);
                        row.ds2n_dg2 = row.dflux_dg2 / ferr
                            - row.flux * dferr_ds * dsig2 / (ferr * ferr);
                        AfpfsCatRow& arow =
                            *reinterpret_cast<AfpfsCatRow*>(row_ptr);
                        arow.sigma = sig_j;
                        arow.dsigma_dg1 = dsig1;
                        arow.dsigma_dg2 = dsig2;
                    }
                }
            }
            return out;
        };

        // Combine nst stacks of ncol measured modes (``buf + k * stride``
        // for stack k) into the interpolated modes and their kernel-size
        // derivative.  The modes fill the FIRST ncol fields of
        // FpfsShapelets (the field order is that of the basis); with 9
        // modes the m60, m64c, m64s slots are zero.  One stack (fixed
        // kernel): the value is the stack itself and the derivative is
        // zero.  Four stacks (adaptive kernel): the Catmull-Rom cubic
        // at t through the four nodes, and its slope times ``dscale`` =
        // dt / dsigma.
        static inline ModesDs
        combine_stacks(
            const double* buf, int stride, int ncol, int nst,
            double t, double dscale
        ) {
            ModesDs out;
            out.v = FpfsShapelets{};
            out.ds = FpfsShapelets{};
            double* v = reinterpret_cast<double*>(&out.v);
            if (nst == 1) {
                std::memcpy(v, buf, sizeof(double) * ncol);
                return out;
            }
            double* ds = reinterpret_cast<double*>(&out.ds);
            double w[adaptive_nst], dw[adaptive_nst];
            catmull_rom(t, w, dw);
            for (int k = 0; k < adaptive_nst; ++k) {
                const double* b = buf + k * stride;
                for (int i = 0; i < ncol; ++i) {
                    v[i] += w[k] * b[i];
                    ds[i] += dw[k] * b[i];
                }
            }
            for (int i = 0; i < ncol; ++i) {
                ds[i] *= dscale;
            }
            return out;
        };

        // One source's modes at (y, x) through nst filters of ncol
        // columns each (one transform of the cutout, nst filter
        // passes): the same operations as the loop body of
        // FpfsImage::measure_source.  Each ``fimg[k]`` is a contiguous
        // (nky, nkx, ncol) deconvolved filter buffer.
        inline ModesDs
        measure_at(
            Image& img_obj,
            const py::array_t<pixel_t>& gal_array,
            const std::complex<double>* const* fimg,
            int ncol,
            int nst,
            double t,
            double dscale,
            double y,
            double x
        ) const {
            const int y_index = static_cast<int>(std::round(y));
            const int x_index = static_cast<int>(std::round(x));
            const double dy = y - y_index;
            const double dx = x - x_index;
            img_obj.set_r(gal_array, x_index, y_index, false);
            img_obj.fft();
            double mm[12 * adaptive_nst];
            for (int k = 0; k < nst; ++k) {
                img_obj.measure_into_raw(fimg[k], ncol, dy, dx, mm + k * ncol);
            }
            for (int i = 0; i < ncol * nst; ++i) {
                mm[i] *= this->fft_ratio;
            }
            return combine_stacks(mm, ncol, ncol, nst, t, dscale);
        };

        // Option-2 noise moments of one source from ONE transform of the
        // pure-noise cutout: the added noise and the two assigned
        // responses (3 ncol modes per filter stack), same operations as
        // measure_at.
        inline void
        measure_noise_spectral(
            Image& img_obj,
            const py::array_t<pixel_t>& noise_array,
            const std::complex<double>* const* add,
            int ncol,
            int nst,
            double t,
            double dscale,
            double y,
            double x,
            ModesDs& xw,
            ModesDs& xk1,
            ModesDs& xk2
        ) const {
            const int y_index = static_cast<int>(std::round(y));
            const int x_index = static_cast<int>(std::round(x));
            const double dy = y - y_index;
            const double dx = x - x_index;
            img_obj.set_r(noise_array, x_index, y_index, false);
            img_obj.fft();
            const int n3 = 3 * ncol;
            double ma[36 * adaptive_nst];
            for (int k = 0; k < nst; ++k) {
                img_obj.measure_into_raw(add[k], n3, dy, dx, ma + k * n3);
            }
            for (int i = 0; i < n3 * nst; ++i) {
                ma[i] *= this->fft_ratio;
            }
            xw = combine_stacks(ma, n3, ncol, nst, t, dscale);
            xk1 = combine_stacks(ma + ncol, n3, ncol, nst, t, dscale);
            xk2 = combine_stacks(ma + 2 * ncol, n3, ncol, nst, t, dscale);
            return;
        };
    };

    void pyExportFpfsForce(py::module_& fpfs);
    } // namespace fpfs
}

#endif // ANACAL_FPFS_FORCE_H
