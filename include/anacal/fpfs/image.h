#ifndef ANACAL_FPFS_IMG_H
#define ANACAL_FPFS_IMG_H


#include "base.h"

#include <vector>

namespace anacal {
    namespace fpfs {

        class FpfsImage {
    private:
        // Preventing copy (implement these if you need copy semantics)
        FpfsImage(const FpfsImage&) = delete;
        FpfsImage& operator=(const FpfsImage&) = delete;
        Image img_obj;
        double fft_ratio;
        const py::array_t<double> psf_array;
        int nx_array, ny_array;
        int nx2, ny2;

        py::array_t<double>
        measure_with_filter(
            const py::array_t<double>& gal_array,
            const py::array_t<std::complex<double>>& filter_fft,
            double y,
            double x
        );
    public:
        double scale = 1.0;
        double sigma_arcsec;
        double klim;
        double sigma_f;
        int nx, ny;
        int npix_overlap, bound;

        FpfsImage(
            int nx,
            int ny,
            double scale,
            double sigma_arcsec,
            double klim,
            const py::array_t<double>& psf_array,
            bool use_estimate=true,
            int npix_overlap=0,
            int bound=0
        );

        py::array_t<double>
        smooth_image(
            const py::array_t<double>& gal_array,
            int xcen,
            int ycen,
            bool do_rotate
        );

        py::array_t<double>
        smooth_image(
            const py::array_t<double>& gal_array,
            int xcen,
            int ycen,
            const std::optional<py::array_t<double>>& noise_array
        );

        void
        find_peaks(
            std::vector<std::tuple<int, int>>& peaks,
            const py::array_t<double>& gal_conv,
            double fthres,
            double pthres,
            double std_m00,
            double v_min,
            double omega_v,
            int xcen,
            int ycen
        );

        py::array_t<Position>
        detect_source(
            py::array_t<double>& gal_array,
            double fthres,
            double pthres,
            double std_m00,
            double v_min,
            double omega_v,
            const std::optional<py::array_t<double>>& noise_array=std::nullopt,
            const std::optional<py::array_t<int16_t>>& mask_array=std::nullopt
        );

        py::array_t<double>
        measure_source(
            const py::array_t<double>& gal_array,
            const py::array_t<std::complex<double>>& filter_image,
            const py::array_t<double>& psf_array,
            const std::optional<py::array_t<Position>>& det=std::nullopt,
            bool do_rotate=false
        );

        py::array_t<double>
        measure_source_at(
            const py::array_t<double>& gal_array,
            const py::array_t<std::complex<double>>& filter_image,
            const py::array_t<double>& psf_array,
            double y,
            double x,
            bool do_rotate=false
        );

        FpfsImage(FpfsImage&& other) noexcept = default;
        FpfsImage& operator=(FpfsImage&& other) noexcept = default;

        ~FpfsImage() = default;
    };

        inline FpfsImage::FpfsImage(
        int nx,
        int ny,
        double scale,
        double sigma_arcsec,
        double klim,
        const py::array_t<double>& psf_array,
        bool use_estimate,
        int npix_overlap,
        int bound
    ) : img_obj(nx, ny, scale, use_estimate), psf_array(psf_array) {
        if ((sigma_arcsec <= 0) || (sigma_arcsec > 5.0)) {
            throw std::runtime_error(
                "FPFS Error: invalid input sigma_arcsec"
            );
        }
        this->nx = nx;
        this->ny = ny;
        this->nx2 = nx / 2;
        this->ny2 = ny / 2;
        this->scale = scale;
        this->sigma_arcsec = sigma_arcsec;
        this->klim = klim;
        this->sigma_f = 1.0 / sigma_arcsec;
        this->fft_ratio = 1.0 / scale / scale;
        if ((npix_overlap % 2 != 0) || (npix_overlap < 0)) {
            throw std::runtime_error(
                "FPFS Error: npix_overlap is not an even number"
            );
        }
        this->npix_overlap = npix_overlap;
        this->bound = bound;
    }

        inline py::array_t<double>
        FpfsImage::smooth_image(
        const py::array_t<double>& img_array,
        int xcen,
        int ycen,
        bool do_rotate
    ) {
        const Gaussian gauss_model(sigma_f);
        // Prepare PSF
        img_obj.set_r(psf_array, true);
        img_obj.fft();
        if (do_rotate) {
            img_obj.rotate90_f();
        }
        const py::array_t<std::complex<double>> parr = img_obj.draw_f();
        // signal
        img_obj.set_r(img_array, xcen, ycen, false);
        img_obj.fft();
        // Deconvolve the PSF
        img_obj.deconvolve(parr, klim);
        // Convolve Gaussian
        img_obj.filter(gauss_model);
        // Back to Real space
        img_obj.ifft();
        py::array_t<double> img_conv = img_obj.draw_r();
        return img_conv;
    }

        inline py::array_t<double>
        FpfsImage::smooth_image(
        const py::array_t<double>& gal_array,
        int xcen,
        int ycen,
        const std::optional<py::array_t<double>>& noise_array
    ) {
        //TODO: use different PSF here
        py::array_t<double> gal_conv = this->smooth_image(
            gal_array,
            xcen,
            ycen,
            false
        );
        if (noise_array.has_value()) {
            py::array_t<double> noise_conv = this->smooth_image(
                *noise_array,
                xcen,
                ycen,
                true
            );
            auto g_r = gal_conv.mutable_unchecked<2>();
            auto n_r = noise_conv.mutable_unchecked<2>();
            for (ssize_t j = 0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i) {
                    g_r(j, i) = g_r(j, i) + n_r(j, i);
                }
            }
        }
        return gal_conv;
    }

        inline void
        FpfsImage::find_peaks(
        std::vector<std::tuple<int, int>>& peaks,
        const py::array_t<double>& gal_conv,
        double fthres,
        double pthres,
        double std_m00,
        double v_min,
        double omega_v,
        int xcen,
        int ycen
    ) {
        // Never use detections that is too close to boundary
        int bound_patch = std::max(this->npix_overlap / 2, 3);
        auto r = gal_conv.unchecked<2>();
        if ((r.shape(0) != this->ny)  || (r.shape(1) != this->nx)) {
            throw std::runtime_error(
                "FPFS Error: convolved image has wrong shape in find_peaks."
            );
        }

        int ymin = ycen - this->ny2;
        int xmin = xcen - this->nx2;

        double fcut = fthres * std_m00;
        double wdet_cut = pthres - fpfs_det_sigma2 - 0.02;

        if (std::fabs(wdet_cut) < 1e-10) {
            wdet_cut = 0.0;
        }
        if (wdet_cut < 0.0) {
            throw std::runtime_error(
                "FPFS Error: The second selection threshold pthres is too small."
            );
        }
        int drmax2 = 1;

        for (int j = bound_patch; j < this->ny - bound_patch; ++j) {
            for (int i = bound_patch; i < this->nx - bound_patch; ++i) {
                double wdet = 1.0;
                double c = r(j, i);
                for (int dj = -1; dj <= 1; dj++) {
                    int dj2 = dj * dj;
                    for (int di = -1; di <= 1; di++) {
                        int dr2 = di * di + dj2;
                        if ((dr2 <= drmax2) && (dr2 != 0)) {
                            double zv = math::ssfunc2(
                                c - r(j + dj, i + di),
                                v_min,
                                omega_v
                            );
                            wdet = wdet * zv;
                        }
                    }
                }
                int y = j + ymin;
                int x = i + xmin;
                bool sel = (
                    (c > fcut) &&
                    (wdet > wdet_cut) &&
                    (y > this->bound) && (y < this->ny_array - this->bound) &&
                    (x > this->bound) && (x < this->nx_array - this->bound)
                );
                if (sel) {
                    peaks.emplace_back(y, x);
                }
            }
        }
    }

        inline py::array_t<Position>
        FpfsImage::detect_source(
        py::array_t<double>& gal_array,
        double fthres,
        double pthres,
        double std_m00,
        double v_min,
        double omega_v,
        const std::optional<py::array_t<double>>& noise_array,
        const std::optional<py::array_t<int16_t>>& mask_array
    ) {

        auto r = gal_array.unchecked<2>();
        if (mask_array.has_value()) {
            // Mask support is not yet implemented.
        }
        this->ny_array = r.shape(0);
        this->nx_array = r.shape(1);

        // Determine number of patches
        // y direction
        int npatch_y = this->ny_array / (this->ny - this->npix_overlap);
        float npatch_y_f = this->ny_array / (this->ny - this->npix_overlap + 0.0);
        if (npatch_y_f > npatch_y) {
            npatch_y = npatch_y + 1;
        }
        int ny2 = npatch_y * (this->ny - this->npix_overlap) + this->npix_overlap;
        int npad_y = (ny2 - this->ny_array) / 2;

        // x direction
        int npatch_x = this->nx_array / (this->nx - this->npix_overlap);
        float npatch_x_f = this->nx_array / (this->nx - this->npix_overlap + 0.0);
        if (npatch_x_f > npatch_x) {
            npatch_x = npatch_x + 1;
        }
        int nx2 = npatch_x * (this->nx - this->npix_overlap) + this->npix_overlap;
        int npad_x = (nx2 - this->nx_array) / 2;

        std::vector<std::tuple<int, int>> peaks;
        // Do detection in each patch
        for (int j = 0; j < npatch_y; ++j) {
            int ycen = (this->ny - this->npix_overlap) * j + this->ny2 - npad_y;
            for (int i = 0; i < npatch_x; ++i) {
                int xcen = (this->nx - this->npix_overlap) * i + this->nx2 - npad_x;
                py::array_t<double> gal_conv = this->smooth_image(
                    gal_array,
                    xcen,
                    ycen,
                    noise_array
                );
                this->find_peaks(
                    peaks,
                    gal_conv,
                    fthres,
                    pthres,
                    std_m00,
                    v_min,
                    omega_v,
                    xcen,
                    ycen
                );
            }
        }

        int nrow = peaks.size();
        py::array_t<Position> detection(nrow);
        auto src_r = detection.mutable_unchecked<1>();

        for (ssize_t j = 0; j < nrow; ++j) {
            const auto& elem = peaks[j];
            src_r(j).y = std::get<0>(elem);
            src_r(j).x = std::get<1>(elem);
        }
        return detection;
    }

        inline py::array_t<double>
        FpfsImage::measure_with_filter(
        const py::array_t<double>& gal_array,
        const py::array_t<std::complex<double>>& filter_fft,
        double y,
        double x
    ) {
        int y_index = static_cast<int>(std::round(y));
        int x_index = static_cast<int>(std::round(x));
        double dy = y - y_index;
        double dx = x - x_index;

        img_obj.set_r(gal_array, x_index, y_index, false);
        img_obj.fft();
        py::array_t<double> row = img_obj.measure(filter_fft, dy, dx);
        auto row_r = row.mutable_unchecked<1>();
        const ssize_t ncol = row.shape(0);
        for (ssize_t i = 0; i < ncol; ++i) {
            row_r(i) *= fft_ratio;
        }
        return row;
    }

        inline py::array_t<double>
        FpfsImage::measure_source(
        const py::array_t<double>& gal_array,
        const py::array_t<std::complex<double>>& filter_image,
        const py::array_t<double>& psf_array,
        const std::optional<py::array_t<Position>>& det,
        bool do_rotate
    ) {
        ssize_t ndim = filter_image.ndim();
        if ( ndim != 3) {
            throw std::runtime_error(
                "FPFS Error: Input filter image must be 3-dimensional."
            );
        }


        img_obj.set_r(psf_array, false);
        img_obj.fft();
        if (do_rotate){
            img_obj.rotate90_f();
        }
        const py::array_t<std::complex<double>> parr = img_obj.draw_f();
        const py::array_t<std::complex<double>> fimg = deconvolve_filter(
            filter_image,
            parr,
            scale,
            klim
        );

        ssize_t ncol = filter_image.shape()[ndim - 1];
        py::array_t<Position> det_default(1);
        auto r = det_default.mutable_unchecked<1>();
        r(0).y = ny / 2; r(0).x = nx / 2;
        const py::array_t<Position>& det_use = det.has_value() ? *det : det_default;
        auto det_r = det_use.unchecked<1>();

        ssize_t nrow = det_use.shape()[0];
        py::array_t<double> src({nrow, ncol});
        auto src_r = src.mutable_unchecked<2>();
        for (ssize_t j = 0; j < nrow; ++j) {
            py::array_t<double> row = this->measure_with_filter(
                gal_array,
                fimg,
                det_r(j).y,
                det_r(j).x
            );
            auto row_r = row.unchecked<1>();
            for (ssize_t i = 0; i < ncol; ++i) {
                src_r(j, i) = row_r(i);
            }
        }
        return src;
    }

        inline py::array_t<double>
        FpfsImage::measure_source_at(
        const py::array_t<double>& gal_array,
        const py::array_t<std::complex<double>>& filter_image,
        const py::array_t<double>& psf_array,
        double y,
        double x,
        bool do_rotate
    ) {
        ssize_t ndim = filter_image.ndim();
        if (ndim != 3) {
            throw std::runtime_error(
                "FPFS Error: Input filter image must be 3-dimensional."
            );
        }

        img_obj.set_r(psf_array, false);
        img_obj.fft();
        if (do_rotate) {
            img_obj.rotate90_f();
        }
        const py::array_t<std::complex<double>> parr = img_obj.draw_f();
        const py::array_t<std::complex<double>> fimg = deconvolve_filter(
            filter_image,
            parr,
            scale,
            klim
        );

        return this->measure_with_filter(gal_array, fimg, y, x);
    }

        void pyExportFpfsImage(py::module_& fpfs);
    } // namespace fpfs
}

#endif // ANACAL_FPFS_IMG_H
