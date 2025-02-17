#ifndef ANACAL_IMAGE_H
#define ANACAL_IMAGE_H

#include "model.h"
#include "math.h"
#include <fftw3.h>

namespace anacal {
inline constexpr double min_deconv_ratio = 1e-5;

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

    void assert_mode(unsigned int condition) const {
        if (condition==0) {
            throw std::runtime_error(
                "Error: The Image object has a wrong mode."
            );
        }
    }
public:
    int ny, nx;
    double scale=1;

    Image(
        int nx,
        int ny,
        double scale,
        bool use_estimate=true,
        unsigned int mode=3
    );

    void set_r(
        const py::array_t<double>&,
        int xcen=-1, int ycen=-1,
        bool ishift=false
    );

    void set_delta_r(bool ishift=false);

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
        double dy=0.0,
        double dx=0.0
    ) const;

    void deconvolve(
        const py::array_t<std::complex<double>>&,
        double
    );

    void deconvolve(const BaseModel&, double);

    void rotate90_f();

    void irotate90_f();

    py::array_t<std::complex<double>> draw_f() const;

    py::array_t<double> draw_r(bool ishift=false) const;

    Image(Image&& other) noexcept = default;
    Image& operator=(Image&& other) noexcept = default;

    ~Image();
};

py::array_t<std::complex<double>>
compute_fft(
    int nx,
    int ny,
    const py::array_t<double>& data_in,
    bool ishift=false
);

py::array_t<std::complex<double>>
deconvolve_filter(
    const py::array_t<std::complex<double>>& filter_image,
    const py::array_t<std::complex<double>>& parr,
    double scale,
    double klim
);


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
                "FPFS Error: invalid input sigma_arcsec"
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
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        int xcen=-1,
        int ycen=-1
    ) {
        // Prepare PSF
        img_obj.set_r(psf_array, -1, -1, true);
        img_obj.fft();
        const py::array_t<std::complex<double>> parr = img_obj.draw_f();

        // signal
        img_obj.set_r(img_array, xcen, ycen, false);
        img_obj.fft();
        // Deconvolve the PSF
        img_obj.deconvolve(parr, klim);
        // Convolve Gaussian
        img_obj.filter(gauss_model);
        py::array_t<std::complex<double>> imgcov_f = img_obj.draw_f();

        std::optional<py::array_t<std::complex<double>>> imgcov_f_n;
        if (noise_array.has_value()){
            img_obj.set_r(psf_array, -1, -1, true);
            img_obj.fft();
            img_obj.rotate90_f();
            const py::array_t<std::complex<double>> parr_n = img_obj.draw_f();

            // signal
            img_obj.set_r(*noise_array, xcen, ycen, false);
            img_obj.fft();
            // Deconvolve the PSF
            img_obj.deconvolve(parr_n, klim);
            // Convolve Gaussian
            img_obj.filter(gauss_model);
            py::array_t<std::complex<double>> tmp_n = img_obj.draw_f();

            auto r = imgcov_f.mutable_unchecked<2>();
            auto r_n = tmp_n.unchecked<2>();
            for (int j = 0; j < ky_length ; ++j) {
                for (int i = 0; i < kx_length ; ++i) {
                    r(j, i) = r(j, i) + r_n(j, i);
                }
            }
            imgcov_f_n = tmp_n;
        }

        std::vector<math::qnumber> result(this->ny * this->nx);
        // v
        {
            img_obj.set_f(imgcov_f);
            img_obj.ifft();
            py::array_t<double> tmp = img_obj.draw_r();
            auto tmp_r = tmp.unchecked<2>();
            for (ssize_t j = 0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i) {
                    ssize_t index = j * this->nx + i;
                    result[index].v = tmp_r(j, i);
                }
            }
        }

        if (imgcov_f_n) {
            auto r = imgcov_f.mutable_unchecked<2>();
            auto r_n = imgcov_f_n.value().unchecked<2>();
            for (int j = 0; j < ky_length ; ++j) {
                for (int i = 0; i < kx_length ; ++i) {
                    r(j, i) = r(j, i) - 2.0 * r_n(j, i);
                }
            }
        }

        // g1
        {
            img_obj.set_f(imgcov_f);
            img_obj.filter(gauss_g1_model);
            img_obj.ifft();
            py::array_t<double> tmp = img_obj.draw_r();
            auto tmp_r = tmp.unchecked<2>();
            for (ssize_t j = 0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i) {
                    ssize_t index = j * this->nx + i;
                    result[index].g1 = tmp_r(j, i);
                }
            }
        }

        // g2
        {
            img_obj.set_f(imgcov_f);
            img_obj.filter(gauss_g2_model);
            img_obj.ifft();
            py::array_t<double> tmp = img_obj.draw_r();
            auto tmp_r = tmp.unchecked<2>();
            for (ssize_t j = 0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i) {
                    ssize_t index = j * this->nx + i;
                    result[index].g2 = tmp_r(j, i);
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
            for (ssize_t j = 0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i) {
                    ssize_t index = j * this->nx + i;
                    result[index].x1 = tmp_r(j, i);
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
            for (ssize_t j = 0; j < this->ny; ++j) {
                for (ssize_t i = 0; i < this->nx; ++i) {
                    ssize_t index = j * this->nx + i;
                    result[index].x2 = tmp_r(j, i);
                }
            }
        }

        // update two shear responses
        for (ssize_t j = 0; j < this->ny; ++j) {
            double y = (j - this->ny2) * this->scale;
            for (ssize_t i = 0; i < this->nx; ++i) {
                ssize_t index = j * this->nx + i;
                double x = (i - this->nx2) * this->scale;
                result[index].g1 = result[index].g1 + (
                    x * result[index].x1 - y * result[index].x2
                );
                result[index].g2 = result[index].g2 + (
                    x * result[index].x2 + y * result[index].x1
                );
            }
        }
        return result;
    };

    py::array_t<double>
    prepare_qnumber_image(
        const py::array_t<double>& img_array,
        const py::array_t<double>& psf_array,
        const std::optional<py::array_t<double>>& noise_array=std::nullopt,
        int xcen=-1,
        int ycen=-1
    ) {
        auto result = py::array_t<double>({5, ny, nx});
        auto r = result.mutable_unchecked<3>();

        std::vector<math::qnumber> qvec = prepare_qnumber_vector(
            img_array,
            psf_array,
            noise_array,
            xcen,
            ycen
        );
        for (ssize_t j = 0; j < this->ny; ++j) {
            for (ssize_t i = 0; i < this->nx; ++i) {
                ssize_t index = j * this->nx + i;
                r(0, j, i) = qvec[index].v;
                r(1, j, i) = qvec[index].g1;
                r(2, j, i) = qvec[index].g2;
                r(3, j, i) = qvec[index].x1;
                r(4, j, i) = qvec[index].x2;
            }
        }
        return result;
    };


    ImageQ(ImageQ&& other) noexcept = default;
    ImageQ& operator=(ImageQ&& other) noexcept = default;

    ~ImageQ() = default;
};

void pyExportImage(py::module& m);
}

#endif // IMAGE_H
