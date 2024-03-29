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
    public:
        double scale = 1.0;
        double sigma_arcsec;
        double klim;
        double sigma_f;
        int nx, ny;
        const py::array_t<double> psf_array;

        FpfsImage(
            int nx,
            int ny,
            double scale,
            double sigma_arcsec,
            double klim,
            const py::array_t<double>& psf_array
        );

        py::array_t<double>
        smooth_image(
            const py::array_t<double>& gal_array,
            const py::array_t<double>& noise_array
        );

        std::vector<std::tuple<int, int, bool>>
        find_peaks(
            const py::array_t<double>& gal_conv,
            double fthres,
            double pthres,
            double pratio,
            double std_m00,
            double std_v,
            int bound
        );

        py::array_t<double>
        measure_sources(
            const py::array_t<double>& gal_array,
            const py::array_t<std::complex<double>>& filter_image,
            const std::vector<std::tuple<int, int, bool>>& det
        );

        std::vector<std::tuple<int, int, bool>>
        detect_sources(
            const py::array_t<double>& gal_array,
            const py::array_t<double>& noise_array,
            double fthres,
            double pthres,
            double pratio,
            double std_m00,
            double std_v,
            int bound
        );


        FpfsImage(FpfsImage&& other) noexcept = default;
        FpfsImage& operator=(FpfsImage&& other) noexcept = default;

        ~FpfsImage();
    };

    void pyExportFpfs(py::module& m);
}

#endif // FPFS_H
