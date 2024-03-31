#include "anacal.h"


namespace anacal {

Image::Image(
    int nx,
    int ny,
    double scale,
    bool use_estimate
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
    nx2 = nx / 2;
    ny2 = ny / 2;
    npixels = nx * ny;
    npixels_f = ny * (nx / 2 + 1);
    kx_length = nx / 2 + 1;
    ky_length = ny;
    dkx = 2.0 * M_PI / nx / scale;
    dky = 2.0 * M_PI / ny / scale;
    xpad = 0;
    ypad = 0;
    unsigned fftw_flag = use_estimate ? FFTW_ESTIMATE : FFTW_MEASURE;

    data_r = (double*) fftw_malloc(sizeof(double) * npixels);
    data_f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npixels_f);
    memset(data_r, 0, sizeof(double) * npixels);
    memset(data_f, 0, sizeof(fftw_complex) * npixels_f);
    plan_forward = fftw_plan_dft_r2c_2d(ny, nx, data_r, data_f, fftw_flag);
    plan_backward = fftw_plan_dft_c2r_2d(ny, nx, data_f, data_r, fftw_flag);
    return;
}


void
Image::set_r (
    const py::array_t<double>& input,
    bool ishift=false
) {
    const ssize_t* shape = input.shape();
    int arr_ny = shape[0];
    int arr_nx = shape[1];
    if (arr_ny > ny) {
        throw std::runtime_error("Error: input array's ny too large");
    }
    if (arr_nx > nx) {
        throw std::runtime_error("Error: input array's nx too large");
    }
    if ((ny > arr_ny) || (nx > arr_nx)) {
        std::fill_n(data_r, ny * nx, 0.0);
    }
    int off_y = (ny - arr_ny) / 2;
    int off_x = (nx - arr_nx) / 2;
    if (ishift) {
        off_y = off_y + ny /2;
        off_x = off_x + nx /2;
    }
    auto r = input.unchecked<2>();
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
Image::set_r (
    const py::array_t<double>& input,
    int x,
    int y
) {
    auto r = input.unchecked<2>();
    ssize_t arr_ny = r.shape(0);
    ssize_t arr_nx = r.shape(1);
    ssize_t ybeg = y - ny2;
    ssize_t yend = ybeg + ny;
    ssize_t xbeg = x - nx2;
    ssize_t xend = xbeg + nx;
    if (
        (xbeg < 0) || (ybeg < 0) ||
        (xend > arr_nx) || (yend > arr_ny)
    ) {
        throw std::runtime_error("Error: Too close to boundary");
    }
    ssize_t index = 0;
    for (ssize_t j = ybeg; j < yend; ++j) {
        for (ssize_t i = xbeg; i < xend; ++i) {
            data_r[index] = r(j, i);
            index += 1;
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
Image::_rotate90_f(int flip) {
    // copy data (fourier space)
    fftw_complex* data = nullptr;
    data = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npixels_f);
    for (int i =0; i < npixels_f; ++i) {
        data[i][0] = data_f[i][0];
        data[i][1] = data_f[i][1];
    }

    // update data
    // upper half
    for (int iy = ny2; iy < ny; ++iy) {
        int xx = iy - ny2;
        for (int ix = 0; ix < kx_length; ++ix) {
            int yy = ny2 - ix;
            int index = (iy + ny2) % ny * kx_length + ix;
            int index2 = (yy + ny2) % ny * kx_length + xx;
            data_f[index][0] = data[index2][0];
            data_f[index][1] = data[index2][1] * flip;
        }
    }
    // lower half
    for (int iy = 0; iy < ny2; ++iy) {
        int xx = ny2 - iy;
        for (int ix = 0; ix < kx_length - 1; ++ix) {
            int yy = ny2 + ix;
            int index = (iy + ny2) % ny * kx_length + ix;
            int index2 = (yy + ny2) % ny * kx_length + xx;
            data_f[index][0] = data[index2][0];
            data_f[index][1] = -data[index2][1] * flip;
        }
    }
    // lower half with ix = kx_length - 1
    int ix = kx_length -1;
    int yy = 0;
    for (int iy = 0; iy < ny2; ++iy) {
        int xx = nx2 - iy;
        int index = (iy + ny2) % ny * kx_length + ix;
        int index2 = (yy + ny2) % ny * kx_length + xx;
        data_f[index][0] = data[index2][0];
        data_f[index][1] = -data[index2][1] * flip;
    }
    fftw_free(data);
    data = nullptr;
}


void
Image::rotate90_f() {
    Image::_rotate90_f(1);
}


void
Image::irotate90_f() {
    Image::_rotate90_f(-1);
}


void
Image::add_image_f(
    const py::array_t<std::complex<double>>& image
) {
    auto r = image.unchecked<2>();
    for (ssize_t j = 0; j < ky_length ; ++j) {
        for (ssize_t i = 0; i < kx_length ; ++i) {
            ssize_t index = j * kx_length + i;
            data_f[index][0] = data_f[index][0] + r(j, i).real();
            data_f[index][1] = data_f[index][1] + r(j, i).imag();
        }
    }
}


void
Image::subtract_image_f(
    const py::array_t<std::complex<double>>& image
) {
    auto r = image.unchecked<2>();
    for (ssize_t j = 0; j < ky_length ; ++j) {
        for (ssize_t i = 0; i < kx_length ; ++i) {
            ssize_t index = j * kx_length + i;
            data_f[index][0] = data_f[index][0] - r(j, i).real();
            data_f[index][1] = data_f[index][1] - r(j, i).imag();
        }
    }
}


void
Image::filter(
    const BaseModel& filter_model
) {
    for (ssize_t iy = 0; iy < ky_length; ++iy) {
        double ky = ((iy < ny2) ? iy : (iy - ny)) * dky ;
        for (ssize_t ix = 0; ix < kx_length; ++ix) {
            ssize_t index = iy * kx_length + ix;
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
    const py::array_t<std::complex<double>>& filter_image
) {
    auto r = filter_image.unchecked<2>();
    for (ssize_t j = 0; j < ky_length ; ++j) {
        for (ssize_t i = 0; i < kx_length ; ++i) {
            int index = j * kx_length + i;
            std::complex<double> val1(data_f[index][0], data_f[index][1]);
            val1 = val1 * r(j, i);
            data_f[index][0] = val1.real();
            data_f[index][1] = val1.imag();
        }
    }
}


py::array_t<double>
Image::measure(
    const py::array_t<std::complex<double>>& filter_image
) const {
    const ssize_t* shape = filter_image.shape();
    ssize_t nz = shape[0];
    ssize_t ny = shape[1];
    ssize_t nx = shape[2];
    if ((ny != ky_length) || (nx != kx_length)) {
        throw std::runtime_error("Error: input filter shape not correct");
    }

    auto src = py::array_t<double>(nz);
    auto s = src.mutable_unchecked<1>();
    for (ssize_t z = 0; z < nz; z++) {
        s(z) = 0.0;
    }

    auto r = filter_image.unchecked<3>();
    for (ssize_t j = 0; j < ky_length; ++j) {
        ssize_t ji = j * kx_length;
        for (ssize_t i = -1; i < 1; ++i) {
            ssize_t index = ji + (i + kx_length) % kx_length;
            std::complex<double> val(data_f[index][0], data_f[index][1]);
            for (ssize_t z = 0; z < nz; ++z) {
                s(z) = s(z) + (r(z, j, i) * val).real();
            }
        }
        for (ssize_t i = 1; i < kx_length - 1; ++i) {
            ssize_t index = ji + i;
            std::complex<double> val(data_f[index][0], data_f[index][1]);
            for (ssize_t z = 0; z < nz; ++z) {
                s(z) = s(z) + (r(z, j, i) * val).real() * 2.0;
            }
        }
    }
    return src;
}


void
Image::deconvolve(
    const BaseModel& psf_model,
    double klim
) {
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


void
Image::deconvolve(
    const py::array_t<std::complex<double>>& psf_image,
    double klim
) {
    double p0 = klim * klim;
    auto rd = psf_image.unchecked<2>();
    for (int j = 0; j < ky_length; ++j) {
        double ky = ((j < ny2) ? j : (j - ny)) * dky ;
        for (int i = 0; i < kx_length; ++i) {
            double kx = i * dkx;
            double r2 = kx * kx + ky * ky;
            int index = j * kx_length + i;
            if (r2 > p0) {
                data_f[index][0] = 0.0;
                data_f[index][1] = 0.0;
            } else {
                std::complex<double> val1(data_f[index][0], data_f[index][1]);
                val1 = val1 / rd(j, i);
                data_f[index][0] = val1.real();
                data_f[index][1] = val1.imag();
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
    if (plan_forward) fftw_destroy_plan(plan_forward);
    if (plan_backward) fftw_destroy_plan(plan_backward);
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
        .def(py::init<int, int, double, bool>(),
            "Initialize the Convolution object using an ndarray",
            py::arg("nx"), py::arg("ny"), py::arg("scale"),
            py::arg("use_estimate")=false
        )
        .def("set_r",
            static_cast<void (Image::*)(const py::array_t<double>&, bool)>(&Image::set_r),
            "Sets up the image in configuration space",
            py::arg("input"),
            py::arg("ishift")=false
        )
        .def("set_r",
            static_cast<void (Image::*)(const py::array_t<double>&, int, int)>(&Image::set_r),
            "Sets up the image in configuration space",
            py::arg("input"),
            py::arg("x"),
            py::arg("y")
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
        .def("rotate90_f", &Image::rotate90_f,
            "Rotates the image by 90 degree anti-clockwise"
        )
        .def("irotate90_f", &Image::rotate90_f,
            "Rotates the image by 90 degree clockwise"
        )
        .def("filter",
            static_cast<void (Image::*)(const BaseModel&)>(&Image::filter),
            "Convolve method with model object",
            py::arg("filter_model")
        )
        .def("filter",
            static_cast<void (Image::*)(const py::array_t<std::complex<double>>&)>
            (&Image::filter),
            "Convolve method with image object",
            py::arg("filter_image")
        )
        .def("measure", &Image::measure,
            "Meausure moments using filter image",
            py::arg("filter_image")
        )
        .def("add_image_f",
            static_cast<void (Image::*)(const py::array_t<std::complex<double>>&)>
            (&Image::add_image_f),
            "Adds image in Fourier space",
            py::arg("image")
        )
        .def("subtract_image_f",
            static_cast<void (Image::*)(const py::array_t<std::complex<double>>&)>
            (&Image::subtract_image_f),
            "Subtracts image in Fourier space",
            py::arg("image")
        )
        .def("deconvolve",
            static_cast<void (Image::*)(
                const py::array_t<std::complex<double>>&, double
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
