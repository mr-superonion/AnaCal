#ifndef CONVOLVE_H
#define CONVOLVE_H

#include "model.h"
#include <fftw3.h>

class Convolve {
private:
    fftw_plan plan_forward, plan_backward;
    double* in;
    fftw_complex* out;
    int ny, nx, ny2, npixels, npixels_f;
    int kx_length, ky_length;
    double dkx, dky;
    double scale=1;
    double norm_factor;
public:

    Convolve();

    // Initialize the Convolution object with data array
    void initialize(
        const py::array_t<double>& input,
        double scale
    );

    void fft (
        const py::array_t<double>& input
    );

    void filter(
        const BaseModel& filtermod,
        const py::array_t<double>& psf
    );

    void filter(
        const BaseModel& filtermod
    );

    py::array_t<double>
    ifft();

    py::array_t<std::complex<double>>
    draw() const;

    void destroy();

    ~Convolve();
};

#endif // CONVOLVE_H
