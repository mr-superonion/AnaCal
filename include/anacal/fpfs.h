#ifndef ANACAL_FPFS_H
#define ANACAL_FPFS_H

#include "image.h"

namespace anacal {
    class FpfsImage {
    private:
        // Preventing copy (implement these if you need copy semantics)
        FpfsImage(const FpfsImage&) = delete;
        FpfsImage& operator=(const FpfsImage&) = delete;
        Image cimg;
        double fft_ratio;
        const py::array_t<double> psf_array;
    public:
        double scale = 1.0;
        double sigma_arcsec;
        double klim;
        double sigma_f;
        int nx, ny;

        FpfsImage(
            int nx,
            int ny,
            double scale,
            double sigma_arcsec,
            double klim,
            const py::array_t<double>& psf_array,
            bool use_estimate=true
        );

        py::array_t<double>
        smooth_image(
            const py::array_t<double>& gal_array,
            const std::optional<py::array_t<double>>& noise_array=std::nullopt
        );

        std::vector<std::tuple<int, int, bool>>
        find_peak(
            const py::array_t<double>& gal_conv,
            double fthres,
            double pthres,
            double pratio,
            double std_m00,
            double std_v,
            int bound
        );

        std::vector<std::tuple<int, int, bool>>
        detect_source(
            const py::array_t<double>& gal_array,
            double fthres,
            double pthres,
            double pratio,
            double std_m00,
            double std_v,
            int bound,
            const std::optional<py::array_t<double>>& noise_array=std::nullopt
        );

        py::array_t<double>
        measure_source(
            const py::array_t<double>& gal_array,
            const py::array_t<std::complex<double>>& filter_image,
            const std::optional<py::array_t<double>>& psf_array=std::nullopt,
            const std::optional<std::vector<std::tuple<int, int, bool>>>& det=std::nullopt,
            bool do_rotate=false
        );

        FpfsImage(FpfsImage&& other) noexcept = default;
        FpfsImage& operator=(FpfsImage&& other) noexcept = default;

        ~FpfsImage();
    };

    void pyExportFpfs(py::module& m);
}

#endif // FPFS_H
