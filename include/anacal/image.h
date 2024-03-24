#ifndef ANACAL_CONVOLVE_H
#define ANACAL_CONVOLVE_H

#include "model.h"
#include <fftw3.h>

namespace anacal {
    class Image {
    private:
        fftw_plan plan_forward = nullptr;
        fftw_plan plan_backward = nullptr;
        int ny2, npixels, npixels_f;
        int kx_length, ky_length;
        double dkx, dky;
        double norm_factor;
    public:
        int ny, nx;
        double scale=1;
        double* data_r = nullptr;
        fftw_complex* data_f = nullptr;

        Image(int, int, double);

        void set_r(const py::array_t<double>&, bool);

        void set_f(const py::array_t<std::complex<double>>&);

        void fft();

        void ifft();

        void filter(const Image&);

        void filter(const BaseModel&);

        void deconvolve(const Image&, double);

        void deconvolve(const BaseModel&, double);

        py::array_t<std::complex<double>> draw_f() const;

        py::array_t<double> draw_r() const;

        ~Image();
    };

    void pyExportImage(py::module& m);
}

#endif // CONVOLVE_H
