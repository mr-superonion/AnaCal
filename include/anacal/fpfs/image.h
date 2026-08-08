#ifndef ANACAL_FPFS_IMG_H
#define ANACAL_FPFS_IMG_H


#include "base.h"

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

        py::array_t<double>
        measure_with_filter(
            const py::array_t<pixel_t>& gal_array,
            const py::array_t<std::complex<double>>& filter_fft,
            double y,
            double x
        );
    public:
        double scale = 1.0;
        double sigma_arcsec;
        double klim;
        int nx, ny;

        FpfsImage(
            int nx,
            int ny,
            double scale,
            double sigma_arcsec,
            double klim,
            const py::array_t<double>& psf_array,
            bool use_estimate=true
        );

        py::array_t<double>
        measure_source(
            const py::array_t<pixel_t>& gal_array,
            const py::array_t<std::complex<double>>& filter_image,
            const py::array_t<double>& psf_array,
            const std::optional<py::array_t<Position>>& det=std::nullopt,
            bool do_rotate=false
        );

        py::array_t<double>
        measure_source_at(
            const py::array_t<pixel_t>& gal_array,
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
        bool use_estimate
    ) : img_obj(nx, ny, scale, use_estimate), psf_array(psf_array) {
        if ((sigma_arcsec <= 0) || (sigma_arcsec > 5.0)) {
            throw std::runtime_error(
                "FPFS Error: invalid input sigma_arcsec"
            );
        }
        this->nx = nx;
        this->ny = ny;
        this->scale = scale;
        this->sigma_arcsec = sigma_arcsec;
        this->klim = klim;
        this->fft_ratio = 1.0 / scale / scale;
    }

    inline py::array_t<double> FpfsImage::measure_with_filter(
        const py::array_t<pixel_t>& gal_array,
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

    inline py::array_t<double> FpfsImage::measure_source(
        const py::array_t<pixel_t>& gal_array,
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

        // Everything above allocates numpy arrays (psf FFT, the
        // deconvolved filter, the output) and so needs the GIL; the
        // per-source loop below only reads the inputs through unchecked
        // accessors and writes into the output buffer, so the GIL is
        // released for its whole duration -- this is what lets callers
        // thread over blocks of sources.  The braced scope ends the
        // release BEFORE the return statement touches src's refcount.
        {
            ScopedGilRelease release;
            for (ssize_t j = 0; j < nrow; ++j) {
                double y = det_r(j).y;
                double x = det_r(j).x;
                int y_index = static_cast<int>(std::round(y));
                int x_index = static_cast<int>(std::round(x));
                double dy = y - y_index;
                double dx = x - x_index;

                img_obj.set_r(gal_array, x_index, y_index, false);
                img_obj.fft();
                double* out = &src_r(j, 0);
                img_obj.measure_into(fimg, dy, dx, out);
                for (ssize_t i = 0; i < ncol; ++i) {
                    out[i] *= fft_ratio;
                }
            }
        }
        return src;
    }

    inline py::array_t<double> FpfsImage::measure_source_at(
        const py::array_t<pixel_t>& gal_array,
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
