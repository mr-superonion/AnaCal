#ifndef ANACAL_IMAGE_H
#define ANACAL_IMAGE_H

#include "table.h"
#include "model.h"
#include "math/tensor.h"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace anacal {
namespace image {

inline constexpr double min_deconv_ratio = 1e-5;

py::array_t<std::complex<double>>
compute_fft(
    int nx,
    int ny,
    const py::array_t<double>& data_in,
    bool ishift = false
);

py::array_t<std::complex<double>>
deconvolve_filter(
    const py::array_t<std::complex<double>>& filter_image,
    const py::array_t<std::complex<double>>& parr,
    double scale,
    double klim
);

class Image {
private:
    fftw_plan plan_forward = nullptr;
    fftw_plan plan_backward = nullptr;
    int nx2, ny2, npixels, npixels_f;
    int kx_length, ky_length;
    double dkx, dky;
    double norm_factor;
    double* data_r = nullptr;
    fftw_complex* data_f = nullptr;

    // Preventing copy (implement these if you need copy semantics)
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    unsigned int mode;

    void _rotate90_f(int);

    inline void assert_mode(unsigned int condition) const {
        if (condition == 0) {
            throw std::runtime_error(
                "Error: The Image object has a wrong mode."
            );
        }
    }

public:
    int ny, nx;
    double scale = 1;

    Image(
        int nx,
        int ny,
        double scale,
        bool use_estimate = true,
        unsigned int mode = 3
    );

    void set_r(
        const py::array_t<double>&,
        int xcen, int,
        bool ishift = false
    );

    void set_r(
        const py::array_t<double>&,
        bool ishift = false
    );

    void set_delta_r(bool ishift = false);

    void set_f(const py::array_t<std::complex<double>>&);

    void set_delta_f();

    void set_noise_f(unsigned int, const py::array_t<double>&);

    void fft();

    void ifft();

    void add_image_f(const py::array_t<std::complex<double>>&);

    void subtract_image_f(const py::array_t<std::complex<double>>&);

    void filter(const py::array_t<std::complex<double>>&);

    void filter(const BaseModel&);

    py::array_t<double> measure(
        const py::array_t<std::complex<double>>&,
        double dy = 0.0,
        double dx = 0.0
    ) const;

    void deconvolve(
        const py::array_t<std::complex<double>>&,
        double
    );

    void deconvolve(const BaseModel&, double);

    void rotate90_f();

    void irotate90_f();

    py::array_t<std::complex<double>> draw_f() const;

    py::array_t<double> draw_r(bool ishift = false) const;

    Image(Image&& other) noexcept = default;
    Image& operator=(Image&& other) noexcept = default;

    ~Image();

    inline void truncate(double xlim, bool ishift) {
        int off_x = ishift ? this->nx2 : 0;
        int off_y = ishift ? this->ny2 : 0;
        double xlim2 = xlim * xlim;

        for (int j = 0; j < this->ny; ++j) {
            int jj = ((j + off_y) % this->ny - this->ny2);
            double y = jj * this->scale;

            for (int i = 0; i < this->nx; ++i) {
                int ii = ((i + off_x) % this->nx - this->nx2);
                double x = ii * this->scale;

                double r2 = x * x + y * y;
                int index = j * this->nx + i;

                if (r2 > xlim2) {
                    this->data_r[index] = 0.0;
                }
            }
        }
        return;
    };

    inline py::array_t<std::complex<double>>
    get_lens_kernel(
        const py::array_t<double>& psf_array,
        const Gaussian & gauss_model,
        double klim
    ) {
        // Prepare PSF
        this->set_r(psf_array, true);
        this->fft();
        const py::array_t<std::complex<double>> parr = this->draw_f();
        // Delta
        this->set_delta_f();
        // Convolve Gaussian
        this->filter(gauss_model);
        // Deconvolve the PSF
        this->deconvolve(parr, klim);
        /* this->ifft(); */
        /* // We truncate the deconvolved Gaussian kernel */
        /* this->truncate(xlim, true); */
        /* this->fft(); */
        return this->draw_f();
    };
};

inline Image::Image(
    int nx,
    int ny,
    double scale,
    bool use_estimate,
    unsigned int mode
) {
    if (ny % 2 != 0) {
        throw std::runtime_error("ny is not divisible by 2");
    }
    if (nx % 2 != 0) {
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
        data_r = static_cast<double*>(fftw_malloc(sizeof(double) * npixels));
        std::memset(data_r, 0, sizeof(double) * npixels);
    }
    if (mode & 2) {
        data_f = static_cast<fftw_complex*>(
            fftw_malloc(sizeof(fftw_complex) * npixels_f)
        );
        std::memset(data_f, 0, sizeof(fftw_complex) * npixels_f);
    }
    if (mode == 3) {
        plan_forward = fftw_plan_dft_r2c_2d(ny, nx, data_r, data_f, fftw_flag);
        plan_backward = fftw_plan_dft_c2r_2d(ny, nx, data_f, data_r, fftw_flag);
    }
    return;
}

inline void
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

inline void
Image::set_r (
    const py::array_t<double>& input,
    bool ishift
) {
    assert_mode(this->mode & 1);
    auto r = input.unchecked<2>();
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

inline void
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

inline void
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

inline void
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

inline void
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

inline void
Image::fft() {
    assert_mode(this->mode == 3);
    fftw_execute(plan_forward);
    return;
}

inline void
Image::ifft() {
    assert_mode(this->mode == 3);
    fftw_execute(plan_backward);
    for (int i = 0; i < npixels; ++i){
        data_r[i] = data_r[i] * this->norm_factor;
    }
    return;
}

inline void
Image::_rotate90_f(int flip) {
    assert_mode(this->mode & 2);
    // copy data (fourier space)
    fftw_complex* data = nullptr;
    data = static_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * npixels_f)
    );
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

inline void
Image::rotate90_f() {
    assert_mode(this->mode & 2);
    this->_rotate90_f(1);
}

inline void
Image::irotate90_f() {
    assert_mode(this->mode & 2);
    this->_rotate90_f(-1);
}

inline void
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

inline void
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

inline void
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

inline void
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

inline py::array_t<double>
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

inline void
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

inline void
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

inline py::array_t<std::complex<double>>
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

inline py::array_t<double>
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

inline Image::~Image() {
    if (plan_forward) fftw_destroy_plan(plan_forward);
    if (plan_backward) fftw_destroy_plan(plan_backward);
    fftw_free(data_r);
    fftw_free(data_f);
    plan_forward = nullptr;
    plan_backward = nullptr;
    data_r = nullptr;
    data_f = nullptr;
}

inline py::array_t<std::complex<double>>
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

inline py::array_t<std::complex<double>>
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

class ImageQ {
private:
    // Preventing copy (implement these if you need copy semantics)
    double klim;
    int nx, ny;
    double scale;
    bool use_estimate;
    Image img_obj;
    double sigma_arcsec;

    const Gaussian gauss_model;
    const GaussianG1 gauss_g1_model;
    const GaussianG2 gauss_g2_model;
    const GaussianX1 gauss_x1_model;
    const GaussianX2 gauss_x2_model;

    int nx2, ny2;
    int kx_length, ky_length;
    ImageQ(const ImageQ&) = delete;
    ImageQ& operator=(const ImageQ&) = delete;
public:
    ImageQ(
        int nx,
        int ny,
        double scale,
        double sigma_arcsec,
        double klim,
        bool use_estimate = true
    ) :
        nx(nx), ny(ny), scale(scale), use_estimate(use_estimate),
        img_obj(nx, ny, scale, use_estimate),
        sigma_arcsec(sigma_arcsec),
        gauss_model(1.0 / sigma_arcsec),
        gauss_g1_model(1.0 / sigma_arcsec),
        gauss_g2_model(1.0 / sigma_arcsec)

    {
        if ((sigma_arcsec <= 0) || (sigma_arcsec > 5.0)) {
            throw std::runtime_error(
                "ImageQ Error: invalid input sigma_arcsec"
            );
        }
        this->nx2 = nx / 2;
        this->ny2 = ny / 2;
        this->kx_length = nx / 2 + 1;
        this->ky_length = ny;
        this->klim = klim;
        return;
    };

    std::vector<math::qnumber>
    prepare_qnumber_vector(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        int xcen,
        int ycen,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {
        const py::array_t<std::complex<double>> karr = img_obj.get_lens_kernel(
            psf_array,
            gauss_model,
            klim
        );
        // signal
        img_obj.set_r(img_array, xcen, ycen, false);
        img_obj.fft();
        img_obj.filter(karr);
        py::array_t<std::complex<double>> imgcov_f = img_obj.draw_f();

        // image is derived from image + noise
        py::array_t<std::complex<double>> imgcov_f_n;
        bool has_noise = noise_array.has_value();
        if (has_noise) {
            img_obj.set_f(karr);
            img_obj.rotate90_f();
            const py::array_t<std::complex<double>> karr_n = img_obj.draw_f();

            // signal
            img_obj.set_r(*noise_array, xcen, ycen, false);
            img_obj.fft();
            img_obj.filter(karr_n);         // Filtering
            imgcov_f_n = img_obj.draw_f();

            auto r = imgcov_f.mutable_unchecked<2>();
            auto r_n = imgcov_f_n.unchecked<2>();
            for (int j = 0; j < this->ky_length ; ++j) {
                for (int i = 0; i < this->kx_length ; ++i) {
                    r(j, i) = r(j, i) + r_n(j, i);
                }
            }
        }

        std::vector<math::qnumber> result(this->ny * this->nx);
        // v
        {
            img_obj.set_f(imgcov_f);
            img_obj.ifft();
            auto tmp_r = img_obj.draw_r().unchecked<2>();
            for (ssize_t j = 0, idx = 0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                    result[idx].v = tmp_r(j, i);
                }
            }
        }

        // shear response from image - noise
        if (has_noise) {
            auto r = imgcov_f.mutable_unchecked<2>();
            auto r_n = imgcov_f_n.unchecked<2>();
            for (int j = 0; j < this->ky_length ; ++j) {
                for (int i = 0; i < this->kx_length ; ++i) {
                    r(j, i) = r(j, i) - 2.0 * r_n(j, i);
                }
            }
        }

        // g1
        {
            img_obj.set_f(imgcov_f);
            img_obj.filter(gauss_g1_model);
            img_obj.ifft();
            auto tmp_r = img_obj.draw_r().unchecked<2>();
            for (ssize_t j = 0, idx=0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                    result[idx].g1 = tmp_r(j, i);
                }
            }
        }

        // g2
        {
            img_obj.set_f(imgcov_f);
            img_obj.filter(gauss_g2_model);
            img_obj.ifft();
            auto tmp_r = img_obj.draw_r().unchecked<2>();
            for (ssize_t j = 0, idx=0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                    result[idx].g2 = tmp_r(j, i);
                }
            }
        }

        // x1
        {
            img_obj.set_f(imgcov_f);
            img_obj.filter(gauss_x1_model);
            img_obj.ifft();
            py::array_t<double> tmp = img_obj.draw_r();
            auto tmp_r = tmp.unchecked<2>();
            for (ssize_t j = 0, idx=0; j < this->ny; ++j) {
                double y = (j - this->ny2) * this->scale;
                for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                    double x = (i - this->nx2) * this->scale;
                    result[idx].g1 = result[idx].g1 + x * tmp_r(j, i);
                    result[idx].g2 = result[idx].g2 + y * tmp_r(j, i);
                    result[idx].x1 = tmp_r(j, i);
                }
            }
        }

         // x2
        {
            img_obj.set_f(imgcov_f);
            img_obj.filter(gauss_x2_model);
            img_obj.ifft();
            py::array_t<double> tmp = img_obj.draw_r();
            auto tmp_r = tmp.unchecked<2>();
            for (ssize_t j = 0, idx=0; j < this->ny; ++j) {
                double y = (j - this->ny2) * this->scale;
                for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                    double x = (i - this->nx2) * this->scale;
                    result[idx].g1 = result[idx].g1 - y * tmp_r(j, i);
                    result[idx].g2 = result[idx].g2 + x * tmp_r(j, i);
                    result[idx].x2 = tmp_r(j, i);
                }
            }
        }
        return result;
    };

    math::qtensor
    prepare_qtensor(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        int xcen,
        int ycen,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {
        auto data = prepare_qnumber_vector(
            img_array,
            psf_array,
            xcen,
            ycen,
            noise_array
        );
        return math::qtensor::from_image(
            std::move(data),
            static_cast<std::size_t>(this->ny),
            static_cast<std::size_t>(this->nx)
        );
    };

    py::array_t<double>
    prepare_qnumber_image(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        int xcen,
        int ycen,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt
    ) {
        auto result = py::array_t<double>({5, ny, nx});
        auto r = result.mutable_unchecked<3>();

        std::vector<math::qnumber> qvec = prepare_qnumber_vector(
            img_array,
            psf_array,
            xcen,
            ycen,
            noise_array
        );
        for (ssize_t j = 0, idx=0; j < this->ny; ++j) {
            for (ssize_t i = 0; i < this->nx; ++i, ++idx) {
                r(0, j, i) = qvec[idx].v;
                r(1, j, i) = qvec[idx].g1;
                r(2, j, i) = qvec[idx].g2;
                r(3, j, i) = qvec[idx].x1;
                r(4, j, i) = qvec[idx].x2;
            }
        }
        return result;
    };


    ImageQ(ImageQ&& other) noexcept = default;
    ImageQ& operator=(ImageQ&& other) noexcept = default;

    ~ImageQ() = default;
};

inline std::vector<math::qnumber>
prepare_data_block(
    const py::array_t<double>& img_array,
    const py::array_t<double>& psf_array,
    double sigma_arcsec,
    const geometry::block & block,
    const std::optional<py::array_t<double>>& noise_array=std::nullopt
){
    ImageQ img_obj(
        block.nx,
        block.ny,
        block.scale,
        sigma_arcsec,
        3.0 / block.scale,      // klim = 3.0 / scale
        true                    // us estimate in FFTW
    );
    return img_obj.prepare_qnumber_vector(
        img_array,
        psf_array,
        block.xcen,
        block.ycen,
        noise_array
    );
};

inline py::array_t<double>
prepare_data_block_image(
    const py::array_t<double>& img_array,
    const py::array_t<double>& psf_array,
    double sigma_arcsec,
    const geometry::block & block,
    const std::optional<py::array_t<double>>& noise_array=std::nullopt
) {
    auto result = py::array_t<double>({5, block.ny, block.nx});
    auto r = result.mutable_unchecked<3>();

    std::vector<math::qnumber> qvec = prepare_data_block(
        img_array, psf_array, sigma_arcsec, block, noise_array
    );
    for (ssize_t j = 0, idx=0; j < block.ny; ++j) {
        for (ssize_t i = 0; i < block.nx; ++i, ++idx) {
            r(0, j, i) = qvec[idx].v;
            r(1, j, i) = qvec[idx].g1;
            r(2, j, i) = qvec[idx].g2;
            r(3, j, i) = qvec[idx].x1;
            r(4, j, i) = qvec[idx].x2;
        }
    }
    return result;
};


inline std::vector<math::qnumber>
prepare_model_block(
    const py::array_t<table::galRow> catalog,
    double sigma_arcsec,
    const geometry::block & block
){
    std::vector<table::galNumber> cat = table::array_to_objlist(
        catalog
    );
    std::size_t n_pix = block.nx * block.ny;
    std::vector<math::qnumber> data_model(n_pix);
    for (const table::galNumber& ss : cat) {
        const ngmix::modelKernelB kernel = ss.model.prepare_modelB(
            block.scale,
            sigma_arcsec
        );
        ss.model.add_to_block(data_model, block, kernel);
    }
    return data_model;
};

inline py::array_t<double>
prepare_model_block_image(
    const py::array_t<table::galRow> catalog,
    double sigma_arcsec,
    const geometry::block & block
) {
    auto result = py::array_t<double>({5, block.ny, block.nx});
    auto r = result.mutable_unchecked<3>();

    std::vector<math::qnumber> qvec = prepare_model_block(
        catalog, sigma_arcsec, block
    );
    for (ssize_t j = 0, idx=0; j < block.ny; ++j) {
        for (ssize_t i = 0; i < block.nx; ++i, ++idx) {
            r(0, j, i) = qvec[idx].v;
            r(1, j, i) = qvec[idx].g1;
            r(2, j, i) = qvec[idx].g2;
            r(3, j, i) = qvec[idx].x1;
            r(4, j, i) = qvec[idx].x2;
        }
    }
    return result;
};


inline double get_smoothed_variance(
    double scale,
    double sigma_arcsec,
    const py::array_t<double>& psf_array,
    double variance
) {
    double variance_sm = 0.0;
    // number of pixels in x and y used to estimated noise variance
    // result is independent on this
    int npix = 64;
    Image img_obj(npix, npix, scale, true);
    {
        // Prepare PSF
        img_obj.set_r(psf_array, true);
        img_obj.fft();
        const py::array_t<std::complex<double>> parr = img_obj.draw_f();
        {
            // white noise
            auto pf = py::array_t<std::complex<double>>({npix, npix / 2 + 1});
            auto pf_r = pf.mutable_unchecked<2>();
            std::complex<double> vv(std::sqrt(variance) / npix, 0.0);
            std::complex<double> sqrt2vv = std::sqrt(2.0) * vv;
            for (ssize_t j = 0; j < npix; ++j) {
                for (ssize_t i = 1; i < npix / 2; ++i) {
                    pf_r(j, i) = sqrt2vv;
                }
                pf_r(j, 0) = vv;
                pf_r(j, npix / 2) = vv;
            }
            img_obj.set_f(pf);
        }
        // Deconvolve the PSF
        img_obj.deconvolve(parr, 3.0 / scale);
    }
    {
        // Convolve Gaussian
        const Gaussian gauss_model(1.0 / sigma_arcsec);
        img_obj.filter(gauss_model);
    }
    {
        const py::array_t<std::complex<double>> pf_dec = img_obj.draw_f();
        auto pfd_r = pf_dec.unchecked<2>();
        for (ssize_t j = 0; j < npix; ++j) {
            for (ssize_t i = 0; i < npix / 2 + 1; ++i) {
                variance_sm += std::norm(pfd_r(j, i));
            }
        }
    }
    return variance_sm;
};


inline void pyExportImage(py::module& m) {
    py::module_ image_mod = m.def_submodule("image", "submodule for convolution");
    image_mod.def(
        "compute_fft", &compute_fft,
        "Compute the FFT of the image",
        py::arg("nx"),
        py::arg("ny"),
        py::arg("data_in"),
        py::arg("ishift")
    );
    image_mod.def(
        "deconvolve_filter", &deconvolve_filter,
        "Deconvolve the filter (defined in Fourier space)",
        py::arg("filter_image"),
        py::arg("parr"),
        py::arg("scale"),
        py::arg("klim")
    );
    image_mod.def(
        "prepare_data_block", &prepare_data_block,
        "prepare the qnumber data in block",
        py::arg("img_array"),
        py::arg("psf_array"),
        py::arg("sigma_arcsec"),
        py::arg("block"),
        py::arg("noise_array")=py::none()
    );
    image_mod.def(
        "prepare_data_block_image", &prepare_data_block_image,
        "prepare the qnumber data in block return image",
        py::arg("img_array"),
        py::arg("psf_array"),
        py::arg("sigma_arcsec"),
        py::arg("block"),
        py::arg("noise_array")=py::none()
    );
    image_mod.def(
        "prepare_model_block", &prepare_model_block,
        "prepare the qnumber model in block",
        py::arg("catalog"),
        py::arg("sigma_arcsec"),
        py::arg("block")
    );
    image_mod.def(
        "prepare_model_block_image", &prepare_model_block_image,
        "prepare the qnumber model in block",
        py::arg("catalog"),
        py::arg("sigma_arcsec"),
        py::arg("block")
    );
    py::class_<Image>(image_mod, "Image")
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
            >(&Image::set_r),
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
            >(&Image::set_r),
            "Sets up the image in configuration space",
            py::arg("input"),
            py::arg("ishift")=false
        )
        .def("set_f", &Image::set_f,
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
    py::class_<ImageQ>(image_mod, "ImageQ")
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
    image_mod.def(
        "get_smoothed_variance", &get_smoothed_variance,
        "get noise variance for smoothed image",
        py::arg("scale"),
        py::arg("sigma_arcsec"),
        py::arg("psf_array"),
        py::arg("variance")
    );
}

} // namespace image
} // namespace anacal

#endif // ANACAL_IMAGE_H
