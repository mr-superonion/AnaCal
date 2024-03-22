#include "convolve.h"


Convolve::Convolve() :
    plan_forward(nullptr), plan_backward(nullptr), in(nullptr), out(nullptr){}

// Initialize the Convolution object with data array
void Convolve::initialize(
    const py::array_t<double>& input,
    double scale
) {
    this->scale = scale;

    // array
    auto info = input.request();
    ny = info.shape[0];
    nx = info.shape[1];
    norm_factor = 1.0 / nx / ny;
    ny2 = ny / 2;
    npixels = nx * ny;
    npixels_f = ny * (nx / 2 + 1);
    kx_length = nx / 2 + 1;
    ky_length = ny;
    dkx = 2.0 * M_PI / nx / scale;
    dky = 2.0 * M_PI / ny / scale;

    in = (double*) fftw_malloc(sizeof(double) * npixels);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npixels_f);
    plan_forward = fftw_plan_dft_r2c_2d(ny, nx, in, out, FFTW_ESTIMATE);
    plan_backward = fftw_plan_dft_c2r_2d(ny, nx, out, in, FFTW_ESTIMATE);
    fft(input);
}

void Convolve::fft (
    const py::array_t<double>& input
) {
    auto r = input.unchecked<2>();
    for (ssize_t j = 0; j < ny; ++j) {
        for (ssize_t i = 0; i < nx; ++i) {
            in[j * nx + i] = r(j, i);
        }
    }
    fftw_execute(plan_forward);
}

void Convolve::filter(
    const BaseModel& filtermod,
    const py::array_t<double>& psf
){
    auto r = psf.unchecked<2>();
    for (int iy = 0; iy < ky_length; ++iy) {
        double ky = (iy < ny2) ? iy * dky : (iy - ny) * dky ;
        for (int ix = 0; ix < kx_length; ++ix) {
            int index = iy * kx_length + ix;
            double kx = ix * dkx;
            std::complex<double> fft_val(out[index][0], out[index][1]);
            std::complex<double> result = (
                fft_val * filtermod.apply(
                    kx=kx, ky=ky
                ) / r(iy, ix)
            );
            out[index][0] = result.real();
            out[index][1] = result.imag();
        }
    }
}

void
Convolve::filter(
    const BaseModel& filtermod
){
    for (int iy = 0; iy < ky_length; ++iy) {
        double ky = (iy < ny2) ? iy * dky : (iy - ny) * dky ;
        for (int ix = 0; ix < kx_length; ++ix) {
            int index = iy * kx_length + ix;
            double kx = ix * dkx;
            std::complex<double> fft_val(out[index][0], out[index][1]);
            std::complex<double> result = fft_val * filtermod.apply(
                kx=kx, ky=ky
            );
            out[index][0] = result.real();
            out[index][1] = result.imag();
        }
    }
}

py::array_t<double>
Convolve::ifft() {
    fftw_execute(plan_backward);
    auto result = py::array_t<double>({ny, nx});
    auto r = result.mutable_unchecked<2>();
    for (ssize_t j = 0; j < ny; ++j) {
        for (ssize_t i = 0; i < nx; ++i) {
            r(j, i) = in[j * nx + i] * norm_factor;
        }
    }
    return result;
}

py::array_t<std::complex<double>>
Convolve::draw() const {
    // Prepare output array
    auto result = py::array_t<std::complex<double>>({ky_length, kx_length});
    auto r = result.mutable_unchecked<2>(); // Accessor
    for (ssize_t j = 0; j < ky_length ; ++j) {
        for (ssize_t i = 0; i < kx_length ; ++i) {
            int index = j * kx_length + i;
            std::complex<double> fft_value(out[index][0], out[index][1]);
            r(j, i) = fft_value;
        }
    }
    return result;
}

void
Convolve::destroy() {
    if (plan_forward) {
        fftw_destroy_plan(plan_forward);
        fftw_destroy_plan(plan_backward);
        fftw_free(in);
        fftw_free(out);
        plan_forward = nullptr;
        plan_backward = nullptr;
        in = nullptr;
        out = nullptr;
    }
}

Convolve::~Convolve() {
    destroy();
}

PYBIND11_MODULE(convolve, m) {
    py::class_<Convolve>(m, "Convolve")
        .def(py::init<>())
        .def("initialize", &Convolve::initialize,
            "Initialize the Convolution object using an ndarray",
            py::arg("input"), py::arg("scale")
        )
        .def("filter",
            static_cast<void (Convolve::*)(
                const BaseModel&, const py::array_t<double>&)>(&Convolve::filter),
            "Filter method with BaseModel and array",
            py::arg("filtermod"), py::arg("psf")
        )
        .def("filter",
            static_cast<void (Convolve::*)(const BaseModel&)>(&Convolve::filter),
            "Filter method with only BaseModel",
            py::arg("filtermod")
        )
        .def("draw", &Convolve::draw,
            "This function draws the real fft outcome"
        )
        .def("ifft", &Convolve::ifft)
        .def("destroy", &Convolve::destroy);
}
