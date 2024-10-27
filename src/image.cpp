#include "anacal.h"


namespace anacal {

Image::Image(
    int nx,
    int ny,
    double scale,
    bool use_estimate,
    unsigned int mode
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
    this->mode = mode;
    // mode = 1: only initialize configuration space
    // mode = 2: only initialize Fourier space
    // mode = 3: initialize both spaces and forward and backward operations

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
    unsigned fftw_flag = use_estimate ? FFTW_ESTIMATE : FFTW_MEASURE;

    if (mode & 1) {
        data_r = (double*) fftw_malloc(sizeof(double) * npixels);
        memset(data_r, 0, sizeof(double) * npixels);
    }
    if (mode & 2) {
        data_f = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npixels_f);
        memset(data_f, 0, sizeof(fftw_complex) * npixels_f);
    }
    if (mode == 3) {
        plan_forward = fftw_plan_dft_r2c_2d(ny, nx, data_r, data_f, fftw_flag);
        plan_backward = fftw_plan_dft_c2r_2d(ny, nx, data_f, data_r, fftw_flag);
    }
    return;
}


void
Image::set_r (
    const py::array_t<double>& input,
    int xcen,
    int ycen,
    bool ishift
) {
    assert_mode(this->mode & 1);
    auto r = input.unchecked<2>();
    int arr_ny = r.shape(0);
    int arr_nx = r.shape(1);
    if (xcen < 0 || xcen > arr_nx) {
        xcen = arr_nx / 2;
    }
    if (ycen < 0 || ycen > arr_ny) {
        ycen = arr_ny / 2;
    }
    int ybeg = ycen - this->ny2;
    int yend = ybeg + this->ny;
    int xbeg = xcen - this->nx2;
    int xend = xbeg + this->nx;
    int off_x = 0;
    int off_y = 0;
    // for the case the beginning or ending point is outside of the image
    if (xbeg < 0) {
        off_x = -xbeg;
        xbeg = 0;
    }
    if (ybeg < 0) {
        off_y = -ybeg;
        ybeg = 0;
    }
    if (xend > arr_nx) xend = arr_nx;
    if (yend > arr_ny) yend = arr_ny;
    // phase shift by half period
    if (ishift) {
        off_y = off_y + this->ny / 2;
        off_x = off_x + this->nx / 2;
    }
    // First fill in the data_r with 0
    std::fill_n(this->data_r, this->ny * this->nx, 0.0);
    // The part has data
    for (int j = ybeg; j < yend; ++j) {
        int jj = (j - ybeg + off_y)  % this->ny;
        for (int i = xbeg; i < xend; ++i) {
            int ii = (i - xbeg + off_x) % this->nx;
            data_r[jj * this->nx + ii] = r(j, i);
        }
    }
    return;
}


void
Image::set_delta_r (bool ishift) {
    std::fill_n(data_r, ny * nx, 0.0);
    if (ishift){
        data_r[0] = 1.0;
    } else {
        int jj = ny / 2;
        int ii = nx / 2;
        data_r[jj * nx + ii] = 1.0;
    }
    return;
}


void
Image::set_f(
    const py::array_t<std::complex<double>>& input
) {
    assert_mode(this->mode & 2);
    const auto* shape = input.shape();
    if ((shape[0] != ky_length) || (shape[1] != kx_length)) {
        throw std::runtime_error("Error: input filter shape not correct");
    }
    auto r = input.unchecked<2>();
    for (int j = 0; j < ky_length ; ++j) {
        int ji = j * kx_length;
        for (int i = 0; i < kx_length ; ++i) {
            int index = ji + i;
            data_f[index][0] = r(j, i).real();
            data_f[index][1] = r(j, i).imag();
        }
    }
    return;
}


void
Image::set_delta_f() {
    assert_mode(this->mode & 2);
    for (int j = 0; j < ky_length; ++j) {
        int ji = j * kx_length;
        for (int i = 0; i < kx_length; ++i) {
            int index = ji + i;
            data_f[index][0] = 1.0;
            data_f[index][1] = 0.0;
        }
    }
}


void
Image::set_noise_f(
    unsigned int seed,
    const py::array_t<double>& correlation
) {

    assert_mode(this->mode & 2);

    py::array_t<std::complex<double>> ps = compute_fft(
        nx,
        ny,
        correlation,
        true
    );
    auto r = ps.unchecked<2>();

    std::mt19937 engine(seed);
    double std_f = std::sqrt(nx * ny / 2.0);
    std::normal_distribution<double> dist(0.0, std_f);
    for (int j = 0; j < ky_length; ++j) {
        int ji = j * kx_length;
        for (int i = 0; i < kx_length; ++i) {
            int index = ji + i;
            double ff = std::sqrt(std::abs(r(j, i)));
            data_f[index][0] = ff * dist(engine);
            data_f[index][1] = ff * dist(engine);
        }
    }

    {
        // k = (0, 0)
        int i = 0;
        int j = 0;
        double ff = std::sqrt(2.0 * std::abs(r(i, j)));
        data_f[0][0] = ff * dist(engine);
        data_f[0][1] = 0.0;

        // k = (0, ny / 2)
        // F(0, ny / 2)  = F(0, -ny / 2)
        // F(0, ny / 2)  = F(0, -ny / 2) *
        i = 0;
        j = ny2;
        ff = std::sqrt(2.0 * std::abs(r(i, j)));
        int index = j * kx_length + i;
        data_f[index][0] = ff * dist(engine);
        data_f[index][1] = 0.0;

        // k = (nx / 2, 0)
        // F(nx / 2, 0)  = F(-nx / 2, 0)
        // F(nx / 2, 0)  = F(-nx / 2, 0) *
        i = nx2;
        j = 0;
        ff = std::sqrt(2.0 * std::abs(r(i, j)));
        index = j * kx_length + i;
        data_f[index][0] = ff * dist(engine);
        data_f[index][1] = 0.0;
    }

    for (int j = 1; j < ny2; ++j) {
        int j2 = -j + ny;
        {
            int i = 0;
            int index = j * kx_length + i;
            int index2 = j2 * kx_length + i;
            data_f[index][0] = data_f[index2][0];
            data_f[index][1] = -data_f[index2][1];
        }

        {
            int i = nx2;
            int index = j * kx_length + i;
            int index2 = j2 * kx_length + i;
            data_f[index][0] = data_f[index2][0];
            data_f[index][1] = -data_f[index2][1];
        }
    }
}


void
Image::fft() {
    assert_mode(this->mode == 3);
    fftw_execute(plan_forward);
    return;
}


void
Image::ifft() {
    assert_mode(this->mode == 3);
    fftw_execute(plan_backward);
    for (int i = 0; i < npixels; ++i){
        data_r[i] = data_r[i] * norm_factor;
    }
    return;
}


void
Image::_rotate90_f(int flip) {
    assert_mode(this->mode & 2);
    // copy data (fourier space)
    fftw_complex* data = nullptr;
    data = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npixels_f);
    for (int i =0; i < npixels_f; ++i) {
        data[i][0] = data_f[i][0];
        data[i][1] = data_f[i][1];
    }

    // update data
    // upper half
    for (int j = ny2; j < ny; ++j) {
        int xx = j - ny2;
        for (int i = 0; i < kx_length; ++i) {
            int yy = ny2 - i;
            int index = (j + ny2) % ny * kx_length + i;
            int index2 = (yy + ny2) % ny * kx_length + xx;
            data_f[index][0] = data[index2][0];
            data_f[index][1] = data[index2][1] * flip;
        }
    }
    // lower half
    for (int j = 0; j < ny2; ++j) {
        int xx = ny2 - j;
        for (int i = 0; i < kx_length - 1; ++i) {
            int yy = ny2 + i;
            int index = (j + ny2) % ny * kx_length + i;
            int index2 = (yy + ny2) % ny * kx_length + xx;
            data_f[index][0] = data[index2][0];
            data_f[index][1] = -data[index2][1] * flip;
        }
    }
    // lower half with i = kx_length - 1
    int i = kx_length -1;
    int yy = 0;
    for (int j = 0; j < ny2; ++j) {
        int xx = nx2 - j;
        int index = (j + ny2) % ny * kx_length + i;
        int index2 = (yy + ny2) % ny * kx_length + xx;
        data_f[index][0] = data[index2][0];
        data_f[index][1] = -data[index2][1] * flip;
    }
    fftw_free(data);
    data = nullptr;
}


void
Image::rotate90_f() {
    assert_mode(this->mode & 2);
    Image::_rotate90_f(1);
}


void
Image::irotate90_f() {
    assert_mode(this->mode & 2);
    Image::_rotate90_f(-1);
}


void
Image::add_image_f(
    const py::array_t<std::complex<double>>& image
) {
    assert_mode(this->mode & 2);
    auto r = image.unchecked<2>();
    for (int j = 0; j < ky_length ; ++j) {
        for (int i = 0; i < kx_length ; ++i) {
            int index = j * kx_length + i;
            data_f[index][0] = data_f[index][0] + r(j, i).real();
            data_f[index][1] = data_f[index][1] + r(j, i).imag();
        }
    }
}


void
Image::subtract_image_f(
    const py::array_t<std::complex<double>>& image
) {
    assert_mode(this->mode & 2);
    auto r = image.unchecked<2>();
    for (int j = 0; j < ky_length ; ++j) {
        for (int i = 0; i < kx_length ; ++i) {
            int index = j * kx_length + i;
            data_f[index][0] = data_f[index][0] - r(j, i).real();
            data_f[index][1] = data_f[index][1] - r(j, i).imag();
        }
    }
}


void
Image::filter(
    const BaseModel& filter_model
) {
    assert_mode(this->mode & 2);
    for (int j = 0; j < ky_length; ++j) {
        double ky = ((j < ny2) ? j : (j - ny)) * dky ;
        for (int i = 0; i < kx_length; ++i) {
            int index = j * kx_length + i;
            double kx = i * dkx;
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
    assert_mode(this->mode & 2);
    auto r = filter_image.unchecked<2>();
    for (int j = 0; j < ky_length ; ++j) {
        for (int i = 0; i < kx_length ; ++i) {
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
    const py::array_t<std::complex<double>>& filter_image,
    double dy,
    double dx
) const {
    assert_mode(this->mode & 2);
    if ((filter_image.shape()[0] != ky_length) ||
        (filter_image.shape()[1] != kx_length)
    ) {
        throw std::runtime_error("Error: input filter shape not correct");
    }
    const double two_pi = 2.0 * M_PI;

    int ncol = filter_image.shape()[2];

    py::array_t<double> meas(ncol);
    auto meas_r = meas.mutable_unchecked<1>();
    for (int z = 0; z < ncol; z++) {
        meas_r(z) = 0.0;
    }

    auto fr = filter_image.unchecked<3>();
    for (int j = 0; j < ky_length; ++j) {
        int ji = j * kx_length;
        double kj = two_pi * (j <= ny / 2 ? j : j - ny) / ny;
        for (int i = -1; i < 1; ++i) {
            int ii = (i + kx_length) % kx_length;
            int index = ji + ii;
            std::complex<double> val(data_f[index][0], data_f[index][1]);
            double ki = two_pi * ii / nx;
            double phase = kj * dy + ki * dx;
            std::complex<double> factor(
                std::cos(phase), std::sin(phase)
            );
            for (int z = 0; z < ncol; ++z) {
                meas_r(z) = meas_r(z) + (fr(j, ii, z) * factor * val).real();
            }
        }
        for (int i = 1; i < kx_length - 1; ++i) {
            int index = ji + i;
            std::complex<double> val(data_f[index][0], data_f[index][1]);
            double ki = two_pi * i / nx;
            double phase = kj * dy + ki * dx;
            std::complex<double> factor(
                std::cos(phase), std::sin(phase)
            );
            for (int z = 0; z < ncol; ++z) {
                meas_r(z) = meas_r(z) + (fr(j, i, z) * factor * val).real() * 2.0;
            }
        }
    }
    return meas;
}


void
Image::deconvolve(
    const BaseModel& psf_model,
    double klim
) {
    assert_mode(this->mode & 2);
    double klim_sq = klim * klim;

    // Test the value at k=0 is real
    std::complex<double> fp_0 = psf_model.apply(0, 0);
    double v_test = fp_0.imag();
    if ((v_test < 0 ? -v_test : v_test) > 1e-10) {
        throw std::runtime_error(
            "Input PSF model is not real in configuration space"
        );
    }
    // minimum value allowed for deconvolution
    double min_deconv_value = min_deconv_ratio * fp_0.real();

    for (int j = 0; j < ky_length; ++j) {
        double ky = ((j < ny2) ? j : (j - ny)) * dky ;
        for (int i = 0; i < kx_length; ++i) {
            double kx = i * dkx;
            double r2 = kx * kx + ky * ky;
            int index = j * kx_length + i;
            if (r2 > klim_sq) {
                data_f[index][0] = 0.0;
                data_f[index][1] = 0.0;
            } else {
                std::complex<double> val(data_f[index][0], data_f[index][1]);
                std::complex<double> fp_k = psf_model.apply(kx, ky);
                double abs_kval = std::abs(fp_k);
                if (abs_kval < min_deconv_value) {
                    data_f[index][0] = val.real() / min_deconv_value;
                    data_f[index][1] = val.imag() / min_deconv_value;
                } else {
                    std::complex<double> result = val / fp_k;
                    data_f[index][0] = result.real();
                    data_f[index][1] = result.imag();
                }
            }
        }
    }
}


void
Image::deconvolve(
    const py::array_t<std::complex<double>>& psf_image,
    double klim
) {
    assert_mode(this->mode & 2);
    double klim_sq = klim * klim;
    auto rd = psf_image.unchecked<2>();

    // Test the value at k=0 is real
    double v_test = rd(0, 0).imag();
    if ((v_test < 0 ? -v_test : v_test) > 1e-10) {
        throw std::runtime_error(
            "Input PSF image is not real in configuration space"
        );
    }
    // minimum value allowed for deconvolution
    double min_deconv_value = min_deconv_ratio * rd(0, 0).real();

    for (int j = 0; j < ky_length; ++j) {
        double ky = ((j < ny2) ? j : (j - ny)) * dky;
        int ji = j * kx_length;
        for (int i = 0; i < kx_length; ++i) {
            double kx = i * dkx;
            double r2 = kx * kx + ky * ky;
            int index = ji + i;
            if (r2 > klim_sq) {
                data_f[index][0] = 0.0;
                data_f[index][1] = 0.0;
            } else {
                std::complex<double> val(data_f[index][0], data_f[index][1]);
                double abs_kval = std::abs(rd(j, i));
                if (abs_kval < min_deconv_value) {
                    data_f[index][0] = val.real() / min_deconv_value;
                    data_f[index][1] = val.imag() / min_deconv_value;
                } else {
                    val = val / rd(j, i);
                    data_f[index][0] = val.real();
                    data_f[index][1] = val.imag();
                }
            }
        }
    }
}


py::array_t<std::complex<double>>
Image::draw_f() const {
    assert_mode(this->mode & 2);
    // Prepare data_fput array
    auto result = py::array_t<std::complex<double>>({ky_length, kx_length});
    auto r = result.mutable_unchecked<2>(); // Accessor
    for (int j = 0; j < ky_length ; ++j) {
        for (int i = 0; i < kx_length ; ++i) {
            int index = j * kx_length + i;
            std::complex<double> val(data_f[index][0], data_f[index][1]);
            r(j, i) = val;
        }
    }
    return result;
}


py::array_t<double>
Image::draw_r(bool ishift) const {
    // ishfit determines whether shift by (ny // 2, nx // 2)
    assert_mode(this->mode & 1);
    auto result = py::array_t<double>({ny, nx});
    auto r = result.mutable_unchecked<2>();
    if (ishift) {
        for (int j = 0; j < ny; ++j) {
            int jj = (j + ny2) % ny;
            int ji = jj * nx;
            for (int i = 0; i < nx; ++i) {
                int ii = (i + nx2) % nx;
                r(j, i) = data_r[ji + ii];
            }
        }
    } else {
        for (int j = 0; j < ny; ++j) {
            int ji = j * nx;
            for (int i = 0; i < nx; ++i) {
                r(j, i) = data_r[ji + i];
            }
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


py::array_t<std::complex<double>>
compute_fft(
    int nx,
    int ny,
    const py::array_t<double>& data_in,
    bool ishift
) {
    Image image(nx, ny, 1.0);
    image.set_r(data_in, ishift);
    image.fft();
    py::array_t<std::complex<double>> data_out = image.draw_f();
    return data_out;
}

py::array_t<std::complex<double>>
deconvolve_filter(
    const py::array_t<std::complex<double>>& filter_image,
    const py::array_t<std::complex<double>>& parr,
    double scale,
    double klim
) {

    int nky = filter_image.shape()[0];
    int nkx = filter_image.shape()[1];

    if (nky % 2 != 0) {
        throw std::runtime_error("nky is not divisible by 2");
    }
    if (parr.shape()[0] != nky) {
        throw std::runtime_error("filter_image and parr have different shape");
    }
    if (parr.shape()[1] != nkx) {
        throw std::runtime_error("filter_image and parr have different shape");
    }

    int ncol = filter_image.shape()[2];
    double dky = 2.0 * M_PI / nky / scale;
    double dkx = 2.0 * M_PI / (2 * (nkx - 1)) / scale;

    double p0 = klim * klim;
    auto f_r = filter_image.unchecked<3>();
    auto p_r = parr.unchecked<2>();

    // Test the value at k=0 is real
    double v_test = p_r(0, 0).imag();
    if ((v_test < 0 ? -v_test : v_test) > 1e-10) {
        throw std::runtime_error(
            "Input PSF image is not real in configuration space"
        );
    }
    // minimum value allowed for deconvolution
    double min_deconv_value = min_deconv_ratio * p_r(0, 0).real();

    py::array_t<std::complex<double>> output({nky, nkx, ncol});
    auto o_r = output.mutable_unchecked<3>();
    for (int j = 0; j < nky; ++j) {
        double ky = ((j < nky / 2) ? j : (j - nky)) * dky ;
        for (int i = 0; i < nkx; ++i) {
            double kx = i * dkx;
            double r2 = kx * kx + ky * ky;
            if (r2 > p0) {
                for (int icol = 0; icol < ncol; icol++) {
                    o_r(j, i, icol) = 0;
                }
            } else {
                std::complex<double> val;
                double abs_kval = std::abs(p_r(j, i));
                if (abs_kval < min_deconv_value) {
                    val = 1.0 / min_deconv_value;
                } else {
                    val = 1.0 / p_r(j, i);
                }
                for (int icol = 0; icol < ncol; icol++) {
                    o_r(j, i, icol) = f_r(j, i, icol) * val;
                }
            }
        }
    }
    return output;
}

void
pyExportImage(py::module& m) {
    py::module_ image = m.def_submodule("image", "submodule for convolution");
    image.def(
        "compute_fft", &compute_fft,
        "Compute the FFT of the image",
        py::arg("nx"),
        py::arg("ny"),
        py::arg("data_in"),
        py::arg("ishift")
    );
    image.def(
        "deconvolve_filter", &deconvolve_filter,
        "Deconvolve the filter (defined in Fourier space)",
        py::arg("filter_image"),
        py::arg("parr"),
        py::arg("scale"),
        py::arg("klim")
    );
    py::class_<Image>(image, "Image")
        .def(py::init<int, int, double, bool, unsigned int>(),
            "Initialize the Convolution object using an ndarray",
            py::arg("nx"), py::arg("ny"), py::arg("scale"),
            py::arg("use_estimate")=true,
            py::arg("mode")=3
        )
        .def("set_r", &Image::set_r,
            "Sets up the image in configuration space",
            py::arg("input"),
            py::arg("xcen")=-1,
            py::arg("ycen")=-1,
            py::arg("ishift")=false
        )
        .def("set_f", &Image::set_f,
            "Sets up the image in Fourier space",
            py::arg("input")
        )
        .def("set_noise_f",
            py::overload_cast<unsigned int, const py::array_t<double>&>
            (&Image::set_noise_f),
            "Sets up noise image in Fourier space using correlation function",
            py::arg("seed"),
            py::arg("correlation")
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
            py::overload_cast<const BaseModel&>
            (&Image::filter),
            "Convolve method with model object",
            py::arg("filter_model")
        )
        .def("filter",
            py::overload_cast<const py::array_t<std::complex<double>>&>
            (&Image::filter),
            "Convolve method with image object",
            py::arg("filter_image")
        )
        .def("measure", &Image::measure,
            "Meausure moments using filter image",
            py::arg("filter_image"),
            py::arg("dy")=0.0,
            py::arg("dx")=0.0
        )
        .def("add_image_f",
            py::overload_cast<const py::array_t<std::complex<double>>&>
            (&Image::add_image_f),
            "Adds image in Fourier space",
            py::arg("image")
        )
        .def("subtract_image_f",
            py::overload_cast<const py::array_t<std::complex<double>>&>
            (&Image::subtract_image_f),
            "Subtracts image in Fourier space",
            py::arg("image")
        )
        .def("deconvolve",
            py::overload_cast<
                const py::array_t<std::complex<double>>&, double
            >(&Image::deconvolve),
            "Defilter method with image object",
            py::arg("psf_image"),
            py::arg("klim")
        )
        .def("deconvolve",
            py::overload_cast<
                const BaseModel&, double
            >(&Image::deconvolve),
            "Defilter method with model object",
            py::arg("psf_model"),
            py::arg("klim")
        )
        .def("draw_r", &Image::draw_r,
            "This function draws the image in configuration space",
            py::arg("ishift")=false
        )
        .def("draw_f", &Image::draw_f,
            "This function draws the image's real fft"
        );
}

}
