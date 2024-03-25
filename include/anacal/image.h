#ifndef ANACAL_IMAGE_H
#define ANACAL_IMAGE_H

#include "model.h"
#include <fftw3.h>

namespace anacal {
    class Image {
    private:
        fftw_plan plan_forward = nullptr;
        fftw_plan plan_backward = nullptr;
        int nx2, ny2, npixels, npixels_f;
        int kx_length, ky_length;
        double dkx, dky;
        double norm_factor;
        int xpad, ypad;
        double* data_r = nullptr;
        fftw_complex* data_f = nullptr;

        // Preventing copy (implement these if you need copy semantics)
        Image(const Image&) = delete;
        Image& operator=(const Image&) = delete;
    public:
        int ny, nx;
        double scale=1;

        Image(int, int, double);

        void set_r(const py::array_t<double>&, bool);

        void set_f(const py::array_t<std::complex<double>>&);

        void fft();

        void ifft();

        const double* view_data_r() const {return data_r;}

        const fftw_complex* view_data_f() const {return data_f;}

        void add_image_f(const Image&);

        void subtract_image_f(const Image&);

        void filter(const Image&);

        void filter(const BaseModel&);

        void deconvolve(const Image&, double);

        void deconvolve(const BaseModel&, double);

        void rotate90_f();

        py::array_t<std::complex<double>> draw_f() const;

        py::array_t<double> draw_r() const;

        Image(Image&& other) noexcept = default;
        Image& operator=(Image&& other) noexcept = default;

        ~Image();
    };

    void pyExportImage(py::module& m);
}

#endif // IMAGE_H
