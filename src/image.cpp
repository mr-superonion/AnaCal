#include "anacal.h"


namespace anacal {

Image::Image(
    int nx,
    int ny,
    double scale
) {
    if (ny % 2 != 0) {
        throw std::runtime_error("ny is not divisible by 2");
    }
    if (nx %2 != 0) {
        throw std::runtime_error("nx is not divisible by 2");
    }

    this->nx = nx;
    this->ny = ny;
    this->scale = scale;

    // array
    norm_factor = 1.0 / nx / ny;
    ny2 = ny / 2;
    npixels = nx * ny;
    npixels_f = ny * (nx / 2 + 1);
    kx_length = nx / 2 + 1;
    ky_length = ny;
    dkx = 2.0 * M_PI / nx / scale;
    dky = 2.0 * M_PI / ny / scale;

    data_r = (double*) fftw_malloc(sizeof(double) * npixels);
    data_f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npixels_f);
    memset(data_r, 0, sizeof(double) * npixels);
    memset(data_f, 0, sizeof(fftw_complex) * npixels_f);
    plan_forward = fftw_plan_dft_r2c_2d(ny, nx, data_r, data_f, FFTW_ESTIMATE);
    plan_backward = fftw_plan_dft_c2r_2d(ny, nx, data_f, data_r, FFTW_ESTIMATE);
    return;
}


void
Image::set_r (
    const py::array_t<double>& input,
    bool ishift=false
) {
    auto shape = input.shape();
    int arr_ny = shape[0];
    int arr_nx = shape[1];
    if (arr_ny > ny) {
        throw std::runtime_error("Error: input array's ny too large");
    }
    if (arr_nx > nx) {
        throw std::runtime_error("Error: input array's nx too large");
    }
    int off_y = (ny - arr_ny) / 2;
    int off_x = (nx - arr_nx) / 2;
    if (ishift) {
        off_y = off_y + ny /2;
        off_x = off_x + nx /2;
    }

    auto r = input.unchecked<2>();
    std::fill_n(data_r, ny * nx, 0.0);
    for (ssize_t j = 0; j < arr_ny; ++j) {
        ssize_t jj = (j + off_y) % ny;
        for (ssize_t i = 0; i < arr_nx; ++i) {
            ssize_t ii = (i + off_x) % nx;
            data_r[jj * nx + ii] = r(j, i);
        }
    }
    return;
}


void
Image::set_f(
    const py::array_t<std::complex<double>>& input
) {
    auto r = input.unchecked<2>();
    for (ssize_t j = 0; j < ky_length ; ++j) {
        for (ssize_t i = 0; i < kx_length ; ++i) {
            int index = j * kx_length + i;
            data_f[index][0] = r(j, i).real();
            data_f[index][1] = r(j, i).imag();
        }
    }
    return;
}


void
Image::fft() {
    fftw_execute(plan_forward);
    return;
}


void
Image::ifft() {
    fftw_execute(plan_backward);
    for (ssize_t i = 0; i < npixels; ++i){
        data_r[i] = data_r[i] * norm_factor;
    }
    return;
}


void
Image::filter(
    const BaseModel& filter_model
){
    for (int iy = 0; iy < ky_length; ++iy) {
        double ky = ((iy < ny2) ? iy : (iy - ny)) * dky ;
        for (int ix = 0; ix < kx_length; ++ix) {
            int index = iy * kx_length + ix;
            double kx = ix * dkx;
            std::complex<double> val(data_f[index][0], data_f[index][1]);
            std::complex<double> result = val * filter_model.apply(kx, ky);
            data_f[index][0] = result.real();
            data_f[index][1] = result.imag();
        }
    }
}


void
Image::filter(
    const Image& filter_image
){
    fftw_complex* pr = filter_image.data_f;
    for (int iy = 0; iy < ky_length; ++iy) {
        for (int ix = 0; ix < kx_length; ++ix) {
            int index = iy * kx_length + ix;
            std::complex<double> val1(data_f[index][0], data_f[index][1]);
            std::complex<double> val2(pr[index][0], pr[index][1]);
            val1 = val1 * val2;
            data_f[index][0] = val1.real();
            data_f[index][1] = val1.imag();
        }
    }
}


void
Image::deconvolve(
    const Image& psf_image,
    double klim
){
    double p0 = klim * klim;
    fftw_complex* pr = psf_image.data_f;
    for (int iy = 0; iy < ky_length; ++iy) {
        double ky = ((iy < ny2) ? iy : (iy - ny)) * dky ;
        for (int ix = 0; ix < kx_length; ++ix) {
            double kx = ix * dkx;
            double r2 = kx * kx + ky * ky;
            int index = iy * kx_length + ix;
            if (r2 > p0) {
                data_f[index][0] = 0.0;
                data_f[index][1] = 0.0;
            } else {
                std::complex<double> val1(data_f[index][0], data_f[index][1]);
                std::complex<double> val2(pr[index][0], pr[index][1]);
                val1 = val1 / val2;
                data_f[index][0] = val1.real();
                data_f[index][1] = val1.imag();
            }
        }
    }
}


void
Image::deconvolve(
    const BaseModel& psf_model,
    double klim
){
    double p0 = klim * klim;
    for (int iy = 0; iy < ky_length; ++iy) {
        double ky = ((iy < ny2) ? iy : (iy - ny)) * dky ;
        for (int ix = 0; ix < kx_length; ++ix) {
            double kx = ix * dkx;
            double r2 = kx * kx + ky * ky;
            int index = iy * kx_length + ix;
            if (r2 > p0) {
                data_f[index][0] = 0.0;
                data_f[index][1] = 0.0;
            } else {
                std::complex<double> val(data_f[index][0], data_f[index][1]);
                std::complex<double> result = val / psf_model.apply(kx, ky);
                data_f[index][0] = result.real();
                data_f[index][1] = result.imag();
            }
        }
    }
}


py::array_t<std::complex<double>>
Image::draw_f() const {
    // Prepare data_fput array
    auto result = py::array_t<std::complex<double>>({ky_length, kx_length});
    auto r = result.mutable_unchecked<2>(); // Accessor
    for (ssize_t j = 0; j < ky_length ; ++j) {
        for (ssize_t i = 0; i < kx_length ; ++i) {
            int index = j * kx_length + i;
            std::complex<double> val(data_f[index][0], data_f[index][1]);
            r(j, i) = val;
        }
    }
    return result;
}


py::array_t<double>
Image::draw_r() const {
    auto result = py::array_t<double>({ny, nx});
    auto r = result.mutable_unchecked<2>();
    for (ssize_t j = 0; j < ny; ++j) {
        for (ssize_t i = 0; i < nx; ++i) {
            r(j, i) = data_r[j * nx + i];
        }
    }
    return result;
}

Image::~Image() {
    fftw_destroy_plan(plan_forward);
    fftw_destroy_plan(plan_backward);
    fftw_free(data_r);
    fftw_free(data_f);
    plan_forward = nullptr;
    plan_backward = nullptr;
    data_r = nullptr;
    data_f = nullptr;
}


void
pyExportImage(py::module& m) {
    py::module_ image = m.def_submodule("image", "submodule for convolution");
    py::class_<Image>(image, "Image")
        .def(py::init<int, int, double>(),
            "Initialize the Convolution object using an ndarray",
            py::arg("nx"), py::arg("ny"), py::arg("scale")
        )
        .def("set_r", &Image::set_r,
            "Sets up the image in configuration space",
            py::arg("input"),
            py::arg("ishift")=false
        )
        .def("set_f", &Image::set_f,
            "Sets up the image in Fourier space",
            py::arg("input")
        )
        .def("fft", &Image::fft,
            "Conducts forward Fourier Trasform"
        )
        .def("ifft", &Image::ifft,
            "Conducts backward Fourier Trasform"
        )
        .def("filter",
            static_cast<void (Image::*)(const BaseModel&)>(&Image::filter),
            "Convolve method with model object",
            py::arg("filter_model")
        )
        .def("filter",
            static_cast<void (Image::*)(const Image&)>(&Image::filter),
            "Convolve method with image object",
            py::arg("filter_image")
        )
        .def("deconvolve",
            static_cast<void (Image::*)(
                const Image&, double
            )>(&Image::deconvolve),
            "Defilter method with image object",
            py::arg("psf_image"),
            py::arg("klim")
        )
        .def("deconvolve",
            static_cast<void (Image::*)(
                const BaseModel&, double
            )>(&Image::deconvolve),
            "Defilter method with model object",
            py::arg("psf_model"),
            py::arg("klim")
        )
        .def("draw_r", &Image::draw_r,
            "This function draws the image in configuration space"
        )
        .def("draw_f", &Image::draw_f,
            "This function draws the image's real fft"
        );
}

}
