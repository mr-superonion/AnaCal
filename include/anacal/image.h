#ifndef ANACAL_IMAGE_H
#define ANACAL_IMAGE_H

#include "model.h"
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

    void pyExportImage(py::module& m);
}

#endif // IMAGE_H
