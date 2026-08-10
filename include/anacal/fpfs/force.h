#ifndef ANACAL_FPFS_FORCE_H
#define ANACAL_FPFS_FORCE_H

#include <cstring>

#include "../psfmodel.h"
#include "catalog.h"

namespace anacal {
    namespace fpfs {

    // Lower bound on the per-kernel flux uncertainty; MUST match
    // FLUX_ERR_MIN in python/anacal/fpfs.py (a noiseless image gives
    // std_m00 == 0, which would make s2n and its shear response infinite).
    inline constexpr double flux_err_min = 1e-8;

    // One finished catalog row for a single shapelet kernel: the
    // FpfsShape fields, then the columns the Python assembly used to
    // append (m00_err and the Gaussian-aperture flux family), in the
    // same order.  Kept a flat all-double standard-layout struct so the
    // FpfsShape prefix can be filled with one memcpy and the numpy
    // dtype is a plain packed f8 record.
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

        py::array
        process_image(
            const py::array_t<pixel_t>& gal_array,
            const py::array_t<double>& psf_array,
            const py::array_t<std::complex<double>>& filter_image,
            const py::array_t<Position>& det,
            double std_m00,
            const py::object& out_dtype,
            const std::optional<py::array_t<pixel_t>>& noise_array=std::nullopt,
            // Per-source mask flags + threshold: sources with
            // mask_value > mask_value_max are SKIPPED -- their output
            // rows are zero-filled, keeping the catalog row-aligned
            // with ``det``.  The psf_invalid_mask_value sentinel (414,
            // "PSF invalid in some band") is EXEMPT from the threshold:
            // each band skips only where its OWN model has no coverage.
            const std::optional<
                py::array_t<int, py::array::c_style>
            >& mask_value=std::nullopt,
            const std::optional<int>& mask_value_max=std::nullopt,
            // Native per-source PSF: stamps are drawn INSIDE the
            // GIL-released loop by the model (no Python per-galaxy
            // drawing), and a source outside the model's coverage is
            // skipped with mask_value set to 414 in place (mask_value
            // must then be a writable int32 array).  ``det`` positions
            // are local image coordinates; the offsets map them into
            // the model's frame (the exposure bbox minimum).
            const std::shared_ptr<psfmodel::PerSourcePsf>& psf_model=nullptr,
            double psf_offset_x=0.0,
            double psf_offset_y=0.0
        ) {
            if (filter_image.ndim() != 3) {
                throw std::runtime_error(
                    "FPFS Error: Input filter image must be 3-dimensional."
                );
            }
            const int nky = static_cast<int>(filter_image.shape(0));
            const int nkx = static_cast<int>(filter_image.shape(1));
            const int ncol = static_cast<int>(filter_image.shape(2));
            if (ncol != 12) {
                throw std::runtime_error(
                    "FPFS Error: expected the 12-mode shapelet basis."
                );
            }
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
                mask_value.has_value() && mask_value_max.has_value()
            );
            if (mask_value.has_value() && mask_value->shape(0) != nrow) {
                throw std::runtime_error(
                    "FPFS Error: mask_value must hold one value per source."
                );
            }
            if (native_psf &&
                !(filter_image.flags() & py::array::c_style)) {
                throw std::runtime_error(
                    "FPFS Error: filter image must be C-contiguous."
                );
            }

            // The caller supplies the FINAL column names; only the packed
            // all-f8 layout of FpfsCatRow is fixed here.
            const py::dtype dt = py::dtype::from_args(out_dtype);
            if (dt.itemsize() != static_cast<ssize_t>(sizeof(FpfsCatRow))) {
                throw std::runtime_error(
                    "FPFS Error: out_dtype does not match the catalog row."
                );
            }
            py::array out(dt, nrow);
            FpfsCatRow* rows = static_cast<FpfsCatRow*>(out.mutable_data());

            ImageLease lease(npix, npix, scale, true);
            Image& img_obj = lease.get();
            const bool has_noise = noise_array.has_value();

            // Constant-PSF filters, prepared exactly as the two
            // FpfsImage::measure_source calls did: data filter from the
            // PSF, noise filter from the PSF rotated 90 degrees in
            // Fourier space.  (For a per-source PSF stack the filters
            // are rebuilt inside the loop instead.)
            py::array_t<std::complex<double>> fimg_g;
            py::array_t<std::complex<double>> fimg_n;
            if (!native_psf) {
                img_obj.set_r(psf_array, false);
                img_obj.fft();
                const py::array_t<std::complex<double>> parr =
                    img_obj.draw_f();
                fimg_g = deconvolve_filter(
                    filter_image, parr, scale, klim
                );
                if (has_noise) {
                    img_obj.set_r(psf_array, false);
                    img_obj.fft();
                    img_obj.rotate90_f();
                    const py::array_t<std::complex<double>> parr_n =
                        img_obj.draw_f();
                    fimg_n = deconvolve_filter(
                        filter_image, parr_n, scale, klim
                    );
                }
            }
            const std::complex<double>* filt_ptr =
                native_psf ? filter_image.data() : nullptr;
            const std::complex<double>* fg_ptr =
                native_psf ? nullptr : fimg_g.data();
            const std::complex<double>* fn_ptr =
                (!native_psf && has_noise) ? fimg_n.data() : nullptr;

            auto det_r = det.unchecked<1>();

            // Same arithmetic as the Python assembly: flux and its
            // responses use the vectorized m00_to_flux (multiply by the
            // unit-flux factor), flux_err the scalar overload.
            const double factor = m00_to_flux(1.0, this->sigma_arcsec);
            const double flux_err = std::max(
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
            int* mv_mut = nullptr;
            if (mask_value.has_value() && native_psf) {
                py::array_t<int, py::array::c_style> mv_arr =
                    *mask_value;
                mv_mut = mv_arr.mutable_data();
            }
            {
                ScopedGilRelease release;
                const int* mv_ptr =
                    mask_value.has_value() ? mask_value->data() : nullptr;
                const int mvmax = do_mask_cut ? *mask_value_max : 0;
                std::vector<std::complex<double>> scratch_g;
                std::vector<std::complex<double>> scratch_n;
                std::vector<double> psf_scratch;
                if (native_psf) {
                    psf_scratch.resize(
                        static_cast<std::size_t>(npix) * npix
                    );
                }
                for (ssize_t j = 0; j < nrow; ++j) {
                    // Threshold cut (when configured).  Zero-filled rows
                    // keep the catalog aligned with the input positions.
                    //
                    // With a per-source PSF model the 414 sentinel is
                    // EXEMPT from the cut: it means "PSF invalid in SOME
                    // band", and THIS band decides for itself via the
                    // contains() check below, so a source flagged by
                    // another band is still measured wherever its own
                    // PSF is valid.  The exemption is deliberately
                    // limited to that mode -- nothing writes the
                    // sentinel without a model, so applying it there
                    // would only let a genuinely masked source (whose
                    // smoothed-mask value happens to be exactly 414)
                    // through the cut.  In per-source mode that
                    // collision remains possible but is the price of an
                    // in-range sentinel.
                    if (do_mask_cut && mv_ptr[j] > mvmax &&
                        !(native_psf &&
                          mv_ptr[j] == psfmodel::psf_invalid_mask_value)) {
                        rows[j] = FpfsCatRow{};
                        continue;
                    }
                    const std::complex<double>* fg = fg_ptr;
                    const std::complex<double>* fn = fn_ptr;
                    if (native_psf) {
                        // Draw the per-source PSF inside the released
                        // loop -- no Python per-galaxy drawing.  A
                        // source outside the model's coverage is the
                        // DM InvalidPsfError case: flag it 414 and
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
                            rows[j] = FpfsCatRow{};
                            continue;
                        }
                        img_obj.set_r_raw(
                            psf_scratch.data(), npix, npix, false
                        );
                        img_obj.fft();
                        const std::vector<std::complex<double>> parr =
                            img_obj.draw_f_vec();
                        deconvolve_filter_into(
                            filt_ptr, parr, scale, klim,
                            nky, nkx, ncol, scratch_g
                        );
                        fg = scratch_g.data();
                        if (has_noise) {
                            img_obj.set_r_raw(
                                psf_scratch.data(), npix, npix, false
                            );
                            img_obj.fft();
                            img_obj.rotate90_f();
                            const std::vector<std::complex<double>> parr_n =
                                img_obj.draw_f_vec();
                            deconvolve_filter_into(
                                filt_ptr, parr_n, scale, klim,
                                nky, nkx, ncol, scratch_n
                            );
                            fn = scratch_n.data();
                        }
                    }
                    FpfsShapelets xg;
                    this->measure_at(
                        img_obj, gal_array, fg,
                        det_r(j).y, det_r(j).x, xg
                    );
                    FpfsShapelets x = xg;
                    std::optional<FpfsShapelets> yn = std::nullopt;
                    if (has_noise) {
                        FpfsShapelets xn;
                        this->measure_at(
                            img_obj, *noise_array, fn,
                            det_r(j).y, det_r(j).x, xn
                        );
                        x = xg + xn;
                        yn = xn;
                    }
                    const FpfsShapeletsResponse dgr =
                        measure_shapelets_dg(x, yn);
                    const FpfsShape sh = measure_fpfs_shape(
                        this->c0, x, dgr
                    );
                    FpfsCatRow& row = rows[j];
                    std::memcpy(&row, &sh, sizeof(FpfsShape));
                    row.m00_err = std_m00;
                    row.flux = sh.m00 * factor;
                    row.dflux_dg1 = sh.dm00_dg1 * factor;
                    row.dflux_dg2 = sh.dm00_dg2 * factor;
                    row.flux_err = flux_err;
                    row.s2n = row.flux / flux_err;
                    row.ds2n_dg1 = row.dflux_dg1 / flux_err;
                    row.ds2n_dg2 = row.dflux_dg2 / flux_err;
                }
            }
            return out;
        };

    private:
        // One source's 12 shapelet modes at (y, x): the same operations
        // as the loop body of FpfsImage::measure_source.  ``fimg`` is a
        // contiguous (nky, nkx, 12) deconvolved filter buffer.
        inline void
        measure_at(
            Image& img_obj,
            const py::array_t<pixel_t>& gal_array,
            const std::complex<double>* fimg,
            double y,
            double x,
            FpfsShapelets& modes
        ) const {
            const int y_index = static_cast<int>(std::round(y));
            const int x_index = static_cast<int>(std::round(x));
            const double dy = y - y_index;
            const double dx = x - x_index;
            img_obj.set_r(gal_array, x_index, y_index, false);
            img_obj.fft();
            double* mm = reinterpret_cast<double*>(&modes);
            img_obj.measure_into_raw(fimg, 12, dy, dx, mm);
            for (int i = 0; i < 12; ++i) {
                mm[i] *= this->fft_ratio;
            }
        };
    };

    void pyExportFpfsForce(py::module_& fpfs);
    } // namespace fpfs
}

#endif // ANACAL_FPFS_FORCE_H
