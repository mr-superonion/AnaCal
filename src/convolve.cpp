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

    Convolve() : plan_forward(nullptr), in(nullptr), out(nullptr), ny(0), nx(0) {}

    // Initialize the Convolution object with data array
    void initialize(
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
        plan_backward = fftw_plan_dft_c2r_2d(ny, ny, out, in, FFTW_ESTIMATE);

        auto r = input.unchecked<2>();
        for (ssize_t j = 0; j < ny; ++j) {
            for (ssize_t i = 0; i < nx; ++i) {
                in[j * nx + i] = r(j, i);
            }
        }
        fftw_execute(plan_forward);
    }

    void filter(
        const BaseFunc& filtermod,
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

    void filter(
        const BaseFunc& filtermod
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
    finalize() {
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
    draw() const {
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

    void destroy() {
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

    ~Convolve() {
        destroy();
    }
};

PYBIND11_MODULE(convolve, m) {
    py::class_<Convolve>(m, "Convolve")
        .def(py::init<>())
        .def("initialize", &Convolve::initialize,
            "Initialize the Convolution object using an ndarray",
            py::arg("input"), py::arg("scale")
        )
        .def("filter",
            static_cast<void (Convolve::*)(
                const BaseFunc&, const py::array_t<double>&)>(&Convolve::filter),
            "Filter method with BaseFunc and array",
            py::arg("filtermod"), py::arg("psf")
        )
        .def("filter",
            static_cast<void (Convolve::*)(const BaseFunc&)>(&Convolve::filter),
            "Filter method with only BaseFunc",
            py::arg("filtermod")
        )
        .def("draw", &Convolve::draw,
            "This function draws the real fft outcome"
        )
        .def("finalize", &Convolve::finalize)
        .def("destroy", &Convolve::destroy);
}
