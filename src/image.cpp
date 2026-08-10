#include "anacal.h"


namespace anacal {

// The FFT engine (pocketfft, see image.h) has no planner and no global
// state, so Image construction and every transform run with no lock at
// all -- FFTW needed the planner serialized because the measurement
// entry points release the GIL and build Images concurrently.

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
    (void) use_estimate;  // pocketfft has no planning modes

    // The buffers start UNINITIALIZED: every setter (set_r/set_r_band/
    // set_delta_r/set_f/set_delta_f/set_noise_f) fully overwrites or
    // re-zeroes its buffer before anything reads it, so a constructor
    // memset would be pure waste (~1 MB per cell-sized image).  Callers
    // must set before they draw/measure.
    if (mode & 1) {
        data_r = static_cast<double*>(::operator new[](
            sizeof(double) * npixels, std::align_val_t(64)
        ));
    }
    if (mode & 2) {
        data_f = static_cast<complex2*>(::operator new[](
            sizeof(complex2) * npixels_f, std::align_val_t(64)
        ));
    }
    return;
}


template <typename T>
void
Image::set_r (
    const py::array_t<T>& input,
    int xcen,
    int ycen,
    bool ishift
) {
    assert_mode(this->mode & 1);
    // ``template`` disambiguator: ``input``'s type depends on T here.
    auto r = input.template unchecked<2>();
    int arr_ny = r.shape(0);
    int arr_nx = r.shape(1);
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
    if (input.flags() & py::array::c_style) {
        // Contiguous input (the normal case): row pointers instead of the
        // stride-carrying unchecked accessor.
        const T* p = input.data();
        for (int j = ybeg; j < yend; ++j) {
            int jj = (j - ybeg + off_y)  % this->ny;
            const T* prow = p + static_cast<std::size_t>(j) * arr_nx;
            for (int i = xbeg; i < xend; ++i) {
                int ii = (i - xbeg + off_x) % this->nx;
                data_r[jj * this->nx + ii] = prow[i];
            }
        }
        return;
    }
    for (int j = ybeg; j < yend; ++j) {
        int jj = (j - ybeg + off_y)  % this->ny;
        for (int i = xbeg; i < xend; ++i) {
            int ii = (i - xbeg + off_x) % this->nx;
            data_r[jj * this->nx + ii] = r(j, i);
        }
    }
    return;
}

template <typename T>
void
Image::set_r (
    const py::array_t<T>& input,
    bool ishift
) {
    assert_mode(this->mode & 1);
    auto r = input.template unchecked<2>();
    int arr_ny = r.shape(0);
    int arr_nx = r.shape(1);
    int xcen = arr_nx / 2;
    int ycen = arr_ny / 2;
    this->set_r(
        input,
        xcen,
        ycen,
        ishift
    );
    return;
}

// The only two pixel types AnaCal reads: float32 for the science and noise
// planes as the surveys store them, double for the PSF stamp and for callers
// that already hold a float64 image.
template void Image::set_r<float>(
    const py::array_t<float>&, int, int, bool
);
template void Image::set_r<double>(
    const py::array_t<double>&, int, int, bool
);
template void Image::set_r<float>(const py::array_t<float>&, bool);
template void Image::set_r<double>(const py::array_t<double>&, bool);


void
Image::set_r_raw (
    const double* data,
    int arr_ny,
    int arr_nx,
    bool ishift
) {
    assert_mode(this->mode & 1);
    const int xcen = arr_nx / 2;
    const int ycen = arr_ny / 2;
    int ybeg = ycen - this->ny2;
    int yend = ybeg + this->ny;
    int xbeg = xcen - this->nx2;
    int xend = xbeg + this->nx;
    int off_x = 0;
    int off_y = 0;
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
    if (ishift) {
        off_y = off_y + this->ny / 2;
        off_x = off_x + this->nx / 2;
    }
    std::fill_n(this->data_r, this->ny * this->nx, 0.0);
    for (int j = ybeg; j < yend; ++j) {
        int jj = (j - ybeg + off_y) % this->ny;
        const double* prow = data + static_cast<std::size_t>(j) * arr_nx;
        for (int i = xbeg; i < xend; ++i) {
            int ii = (i - xbeg + off_x) % this->nx;
            data_r[jj * this->nx + ii] = prow[i];
        }
    }
    return;
}


template <typename T>
void
Image::set_r_band (
    const py::array_t<T>& stack,
    ssize_t band,
    int xcen,
    int ycen,
    bool ishift
) {
    assert_mode(this->mode & 1);
    if (stack.ndim() == 2) {
        if (band != 0) {
            throw std::runtime_error(
                "Image Error: band index out of range for a 2-D image"
            );
        }
        this->set_r(stack, xcen, ycen, ishift);
        return;
    }
    if (stack.ndim() != 3) {
        throw std::runtime_error(
            "Image Error: expected a 2-D image or 3-D (nband, ny, nx) stack"
        );
    }
    if (band < 0 || band >= stack.shape(0)) {
        throw std::runtime_error("Image Error: band index out of range");
    }
    auto r = stack.template unchecked<3>();
    int arr_ny = static_cast<int>(r.shape(1));
    int arr_nx = static_cast<int>(r.shape(2));
    int ybeg = ycen - this->ny2;
    int yend = ybeg + this->ny;
    int xbeg = xcen - this->nx2;
    int xend = xbeg + this->nx;
    int off_x = 0;
    int off_y = 0;
    // Same boundary and phase-shift handling as the 2-D set_r above.
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
    if (ishift) {
        off_y = off_y + this->ny / 2;
        off_x = off_x + this->nx / 2;
    }

    std::fill_n(this->data_r, this->ny * this->nx, 0.0);
    if (stack.flags() & py::array::c_style) {
        // Contiguous stack (the normal case): row pointers instead of the
        // stride-carrying unchecked accessor.
        const T* p = stack.data() + (
            static_cast<std::size_t>(band) * arr_ny * arr_nx
        );
        for (int j = ybeg; j < yend; ++j) {
            int jj = (j - ybeg + off_y)  % this->ny;
            const T* prow = p + static_cast<std::size_t>(j) * arr_nx;
            for (int i = xbeg; i < xend; ++i) {
                int ii = (i - xbeg + off_x) % this->nx;
                data_r[jj * this->nx + ii] = prow[i];
            }
        }
        return;
    }
    for (int j = ybeg; j < yend; ++j) {
        int jj = (j - ybeg + off_y)  % this->ny;
        for (int i = xbeg; i < xend; ++i) {
            int ii = (i - xbeg + off_x) % this->nx;
            data_r[jj * this->nx + ii] = r(band, j, i);
        }
    }
    return;
}

template <typename T>
void
Image::set_r_band (
    const py::array_t<T>& stack,
    ssize_t band,
    bool ishift
) {
    const ssize_t nd = stack.ndim();
    int xcen = static_cast<int>(stack.shape(nd - 1)) / 2;
    int ycen = static_cast<int>(stack.shape(nd - 2)) / 2;
    this->set_r_band(stack, band, xcen, ycen, ishift);
    return;
}

template void Image::set_r_band<float>(
    const py::array_t<float>&, ssize_t, int, int, bool
);
template void Image::set_r_band<double>(
    const py::array_t<double>&, ssize_t, int, int, bool
);
template void Image::set_r_band<float>(
    const py::array_t<float>&, ssize_t, bool
);
template void Image::set_r_band<double>(
    const py::array_t<double>&, ssize_t, bool
);



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
Image::set_f(
    const std::vector<std::complex<double>>& input
) {
    assert_mode(this->mode & 2);
    if (static_cast<int>(input.size()) != ky_length * kx_length) {
        throw std::runtime_error("Error: input filter shape not correct");
    }
    for (int j = 0; j < ky_length ; ++j) {
        int ji = j * kx_length;
        for (int i = 0; i < kx_length ; ++i) {
            int index = ji + i;
            data_f[index][0] = input[index].real();
            data_f[index][1] = input[index].imag();
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
        double ff = std::sqrt(2.0 * std::abs(r(0, 0)));
        data_f[0][0] = ff * dist(engine);
        data_f[0][1] = 0.0;

        // k = (0, ny / 2)
        // F(0, ny / 2)  = F(0, -ny / 2)
        // F(0, ny / 2)  = F(0, -ny / 2) *
        int i = 0;
        int j = ny2;
        ff = std::sqrt(2.0 * std::abs(r(j, i)));
        int index = j * kx_length + i;
        data_f[index][0] = ff * dist(engine);
        data_f[index][1] = 0.0;

        // k = (nx / 2, 0)
        // F(nx / 2, 0)  = F(-nx / 2, 0)
        // F(nx / 2, 0)  = F(-nx / 2, 0) *
        i = nx2;
        j = 0;
        ff = std::sqrt(2.0 * std::abs(r(j, i)));
        index = j * kx_length + i;
        data_f[index][0] = ff * dist(engine);
        data_f[index][1] = 0.0;
    }

    if (nx % 2 == 0 && ny % 2 == 0) {
        int i = nx2;
        int j = ny2;
        int index = j * kx_length + i;
        double ff = std::sqrt(2.0 * std::abs(r(j, i)));
        data_f[index][0] = ff * dist(engine);
        data_f[index][1] = 0.0;
    }

    for (int j = 1; j < ny2; ++j) {
        int j2 = -j + ny;  // -j mod ny
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


// pocketfft argument blocks for this image's 2-D r2c / c2r, matching
// FFTW's layout (real (ny, nx) <-> complex (ny, nx/2+1), row-major)
// and its UNNORMALIZED convention.
void
Image::_fft_shapes(
    pocketfft::shape_t& shape, pocketfft::stride_t& s_real,
    pocketfft::stride_t& s_cplx, pocketfft::shape_t& axes
) const {
    shape = {static_cast<std::size_t>(ny), static_cast<std::size_t>(nx)};
    s_real = {
        static_cast<ptrdiff_t>(sizeof(double) * nx),
        static_cast<ptrdiff_t>(sizeof(double))
    };
    s_cplx = {
        static_cast<ptrdiff_t>(sizeof(complex2) * kx_length),
        static_cast<ptrdiff_t>(sizeof(complex2))
    };
    axes = {0, 1};
}


void
Image::fft() {
    assert_mode(this->mode == 3);
    pocketfft::shape_t shape, axes;
    pocketfft::stride_t s_real, s_cplx;
    _fft_shapes(shape, s_real, s_cplx, axes);
    pocketfft::r2c(
        shape, s_real, s_cplx, axes, pocketfft::FORWARD, data_r,
        reinterpret_cast<std::complex<double>*>(data_f), 1.0, 1
    );
    return;
}


void
Image::ifft() {
    assert_mode(this->mode == 3);
    ifft_raw();
    for (int i = 0; i < npixels; ++i){
        data_r[i] = data_r[i] * this->norm_factor;
    }
    return;
}


void
Image::ifft_raw() {
    assert_mode(this->mode == 3);
    pocketfft::shape_t shape, axes;
    pocketfft::stride_t s_real, s_cplx;
    _fft_shapes(shape, s_real, s_cplx, axes);
    pocketfft::c2r(
        shape, s_cplx, s_real, axes, pocketfft::BACKWARD,
        reinterpret_cast<const std::complex<double>*>(data_f), data_r,
        1.0, 1
    );
    return;
}


void
Image::_rotate90_f(int flip) {
    assert_mode(this->mode & 2);
    // copy data (fourier space) into the persistent scratch buffer
    if (rot_scratch_.size() < static_cast<std::size_t>(npixels_f)) {
        rot_scratch_.resize(npixels_f);
    }
    std::complex<double>* data = rot_scratch_.data();
    for (int i =0; i < npixels_f; ++i) {
        data[i] = std::complex<double>(data_f[i][0], data_f[i][1]);
    }

    // update data
    // upper half
    for (int j = ny2; j < ny; ++j) {
        int xx = j - ny2;
        for (int i = 0; i < kx_length; ++i) {
            int yy = ny2 - i;
            int index = (j + ny2) % ny * kx_length + i;
            int index2 = (yy + ny2) % ny * kx_length + xx;
            data_f[index][0] = data[index2].real();
            data_f[index][1] = data[index2].imag() * flip;
        }
    }
    // lower half
    for (int j = 0; j < ny2; ++j) {
        int xx = ny2 - j;
        for (int i = 0; i < kx_length - 1; ++i) {
            int yy = ny2 + i;
            int index = (j + ny2) % ny * kx_length + i;
            int index2 = (yy + ny2) % ny * kx_length + xx;
            data_f[index][0] = data[index2].real();
            data_f[index][1] = -data[index2].imag() * flip;
        }
    }
    // lower half with i = kx_length - 1
    int i = kx_length -1;
    int yy = 0;
    for (int j = 0; j < ny2; ++j) {
        int xx = nx2 - j;
        int index = (j + ny2) % ny * kx_length + i;
        int index2 = (yy + ny2) % ny * kx_length + xx;
        data_f[index][0] = data[index2].real();
        data_f[index][1] = -data[index2].imag() * flip;
    }
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


void
Image::filter(
    const std::vector<std::complex<double>>& filter_image
) {
    assert_mode(this->mode & 2);
    if (static_cast<int>(filter_image.size()) != ky_length * kx_length) {
        throw std::runtime_error("Error: input filter shape not correct");
    }
    for (int j = 0; j < ky_length ; ++j) {
        for (int i = 0; i < kx_length ; ++i) {
            int index = j * kx_length + i;
            std::complex<double> val1(data_f[index][0], data_f[index][1]);
            val1 = val1 * filter_image[index];
            data_f[index][0] = val1.real();
            data_f[index][1] = val1.imag();
        }
    }
}


void
Image::measure_into(
    const py::array_t<std::complex<double>>& filter_image,
    double dy,
    double dx,
    double* out
) const {
    // Same computation as measure() below, but writing into a
    // caller-provided buffer so it can run with the GIL released
    // (the filter is only READ, through raw pointers / unchecked
    // accessors).
    assert_mode(this->mode & 2);
    if ((filter_image.shape()[0] != ky_length) ||
        (filter_image.shape()[1] != kx_length)
    ) {
        throw std::runtime_error("Error: input filter shape not correct");
    }
    const double two_pi = 2.0 * M_PI;

    int ncol = filter_image.shape()[2];

    if (filter_image.flags() & py::array::c_style) {
        // Contiguous filter (the normal case): walk it with a flat
        // pointer so the innermost z loop is stride-1 and vectorizable.
        this->measure_into_raw(filter_image.data(), ncol, dy, dx, out);
        return;
    }

    for (int z = 0; z < ncol; z++) {
        out[z] = 0.0;
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
                out[z] = out[z] + (fr(j, ii, z) * factor * val).real();
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
                out[z] = out[z] + (fr(j, i, z) * factor * val).real() * 2.0;
            }
        }
    }
    return;
}


void
Image::measure_into_raw(
    const std::complex<double>* fp,
    int ncol,
    double dy,
    double dx,
    double* out
) const {
    assert_mode(this->mode & 2);
    const double two_pi = 2.0 * M_PI;
    for (int z = 0; z < ncol; z++) {
        out[z] = 0.0;
    }
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
            const std::complex<double>* frow = fp + (
                static_cast<std::size_t>(index) * ncol
            );
            for (int z = 0; z < ncol; ++z) {
                out[z] = out[z] + (frow[z] * factor * val).real();
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
            const std::complex<double>* frow = fp + (
                static_cast<std::size_t>(index) * ncol
            );
            for (int z = 0; z < ncol; ++z) {
                out[z] = out[z] + (frow[z] * factor * val).real() * 2.0;
            }
        }
    }
    return;
}


py::array_t<double>
Image::measure(
    const py::array_t<std::complex<double>>& filter_image,
    double dy,
    double dx
) const {
    if (filter_image.ndim() != 3) {
        throw std::runtime_error("Error: input filter shape not correct");
    }
    const int ncol = static_cast<int>(filter_image.shape()[2]);
    py::array_t<double> meas(ncol);
    this->measure_into(filter_image, dy, dx, meas.mutable_data());
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


void
Image::deconvolve(
    const std::vector<std::complex<double>>& psf_image,
    double klim
) {
    assert_mode(this->mode & 2);
    if (static_cast<int>(psf_image.size()) != ky_length * kx_length) {
        throw std::runtime_error("Error: input filter shape not correct");
    }
    double klim_sq = klim * klim;

    // Test the value at k=0 is real
    double v_test = psf_image[0].imag();
    if ((v_test < 0 ? -v_test : v_test) > 1e-10) {
        throw std::runtime_error(
            "Input PSF image is not real in configuration space"
        );
    }
    // minimum value allowed for deconvolution
    double min_deconv_value = min_deconv_ratio * psf_image[0].real();

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
                double abs_kval = std::abs(psf_image[index]);
                if (abs_kval < min_deconv_value) {
                    data_f[index][0] = val.real() / min_deconv_value;
                    data_f[index][1] = val.imag() / min_deconv_value;
                } else {
                    val = val / psf_image[index];
                    data_f[index][0] = val.real();
                    data_f[index][1] = val.imag();
                }
            }
        }
    }
}


std::vector<std::complex<double>>
Image::draw_f_vec() const {
    assert_mode(this->mode & 2);
    std::vector<std::complex<double>> result(ky_length * kx_length);
    for (int j = 0; j < ky_length ; ++j) {
        for (int i = 0; i < kx_length ; ++i) {
            int index = j * kx_length + i;
            result[index] = std::complex<double>(
                data_f[index][0], data_f[index][1]
            );
        }
    }
    return result;
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


Image::Image(Image&& other) noexcept
    : nx2(other.nx2),
      ny2(other.ny2),
      npixels(other.npixels),
      npixels_f(other.npixels_f),
      kx_length(other.kx_length),
      ky_length(other.ky_length),
      dkx(other.dkx),
      dky(other.dky),
      norm_factor(other.norm_factor),
      data_r(other.data_r),
      data_f(other.data_f),
      rot_scratch_(std::move(other.rot_scratch_)),
      mode(other.mode),
      ny(other.ny),
      nx(other.nx),
      scale(other.scale) {
    other.data_r = nullptr;
    other.data_f = nullptr;
    other.nx2 = 0;
    other.ny2 = 0;
    other.npixels = 0;
    other.npixels_f = 0;
    other.kx_length = 0;
    other.ky_length = 0;
    other.dkx = 0.0;
    other.dky = 0.0;
    other.norm_factor = 0.0;
    other.mode = 0;
    other.ny = 0;
    other.nx = 0;
    other.scale = 0.0;
}


Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        _free_buffers();

        nx2 = other.nx2;
        ny2 = other.ny2;
        npixels = other.npixels;
        npixels_f = other.npixels_f;
        kx_length = other.kx_length;
        ky_length = other.ky_length;
        dkx = other.dkx;
        dky = other.dky;
        norm_factor = other.norm_factor;
        data_r = other.data_r;
        data_f = other.data_f;
        rot_scratch_ = std::move(other.rot_scratch_);
        mode = other.mode;
        ny = other.ny;
        nx = other.nx;
        scale = other.scale;

        other.data_r = nullptr;
        other.data_f = nullptr;
        other.nx2 = 0;
        other.ny2 = 0;
        other.npixels = 0;
        other.npixels_f = 0;
        other.kx_length = 0;
        other.ky_length = 0;
        other.dkx = 0.0;
        other.dky = 0.0;
        other.norm_factor = 0.0;
        other.mode = 0;
        other.ny = 0;
        other.nx = 0;
        other.scale = 0.0;
    }
    return *this;
}


void
Image::_free_buffers() noexcept {
    if (data_r != nullptr) {
        ::operator delete[](data_r, std::align_val_t(64));
        data_r = nullptr;
    }
    if (data_f != nullptr) {
        ::operator delete[](data_f, std::align_val_t(64));
        data_f = nullptr;
    }
}


Image::~Image() {
    _free_buffers();
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
    image.def(
        "prepare_data_cell", &prepare_data_cell,
        "prepare the qnumber data in cell",
        py::arg("img_array"),
        py::arg("psf_array"),
        py::arg("sigma_arcsec"),
        py::arg("cell"),
        py::arg("noise_array")=py::none(),
        py::arg("band")=0
    );
    image.def(
        "prepare_data_cell_image", &prepare_data_cell_image,
        "prepare the qnumber data in cell return image",
        py::arg("img_array"),
        py::arg("psf_array"),
        py::arg("sigma_arcsec"),
        py::arg("cell"),
        py::arg("noise_array")=py::none()
    );
    image.def(
        "prepare_model_cell", &prepare_model_cell,
        "prepare the qnumber model in cell",
        py::arg("catalog"),
        py::arg("sigma_arcsec"),
        py::arg("cell")
    );
    image.def(
        "prepare_model_cell_image", &prepare_model_cell_image,
        "prepare the qnumber model in cell",
        py::arg("catalog"),
        py::arg("sigma_arcsec"),
        py::arg("cell")
    );
    py::class_<Image>(image, "Image")
        .def(py::init<int, int, double, bool, unsigned int>(),
            "Initialize the Convolution object using an ndarray",
            py::arg("nx"), py::arg("ny"), py::arg("scale"),
            py::arg("use_estimate")=true,
            py::arg("mode")=3
        )
        .def("set_r",
            py::overload_cast<
                const py::array_t<double>&,
                int,
                int,
                bool
            >
            // Only the float64 form is exposed to Python: this is a low-level
            // utility and its callers already hold float64 arrays.  The float32
            // instantiation is used from C++, on the science and noise planes.
            (&Image::set_r<double>),
            "Sets up the image in configuration space",
            py::arg("input"),
            py::arg("xcen"),
            py::arg("ycen"),
            py::arg("ishift")=false
        )
        .def("set_r",
            py::overload_cast<
                const py::array_t<double>&,
                bool
            >
            (&Image::set_r<double>),
            "Sets up the image in configuration space (force center)",
            py::arg("input"),
            py::arg("ishift")=false
        )
        .def("set_f",
            static_cast<void (Image::*)(
                const py::array_t<std::complex<double>>&
            )>(&Image::set_f),
            "Sets up the image in Fourier space",
            py::arg("input")
        )
        .def("set_delta_r", &Image::set_delta_r,
            "Sets up the delta image in configuration space",
            py::arg("ishift")=false
        )
        .def("set_delta_f", &Image::set_delta_f,
            "Sets up the delta image in Fourier space"
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
        .def("irotate90_f", &Image::irotate90_f,
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
    py::class_<ImageQ>(image, "ImageQ")
        .def(py::init<
                int, int, double, double, double, bool
            >(),
            "Initialize the ImageQ object using an ndarray",
            py::arg("nx"), py::arg("ny"),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("klim"),
            py::arg("use_estimate")=true
        )
        .def("prepare_qnumber_image",
            &ImageQ::prepare_qnumber_image,
            "prepare the qnumber image",
            py::arg("gal_array"),
            py::arg("psf_array"),
            py::arg("xcen"),
            py::arg("ycen"),
            py::arg("noise_array")=py::none()
        )
        .def(
            "prepare_qtensor",
            &ImageQ::prepare_qtensor,
            "Prepare a qtensor view over the qnumber image",
            py::arg("gal_array"),
            py::arg("psf_array"),
            py::arg("xcen"),
            py::arg("ycen"),
            py::arg("noise_array")=py::none()
        );
    image.def(
        "get_smoothed_variance",
        static_cast<double (*)(
            double, double, const py::array_t<double>&, double
        )>(&get_smoothed_variance),
        "get noise variance for smoothed image",
        py::arg("scale"),
        py::arg("sigma_arcsec"),
        py::arg("psf_array"),
        py::arg("variance")
    );
}
}
