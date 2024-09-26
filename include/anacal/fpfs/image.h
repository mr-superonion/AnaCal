#ifndef ANACAL_FPFS_IMG_H
#define ANACAL_FPFS_IMG_H


#include "base.h"

namespace anacal {

    class FpfsImage {
    private:
        // Preventing copy (implement these if you need copy semantics)
        FpfsImage(const FpfsImage&) = delete;
        FpfsImage& operator=(const FpfsImage&) = delete;
        Image img_obj;
        double fft_ratio;
        const py::array_t<double> psf_array;
        int nx_array, ny_array;
        int nx2, ny2;
    public:
        double scale = 1.0;
        double sigma_arcsec;
        double klim;
        double sigma_f;
        int nx, ny;
        int npix_overlap, bound;

        FpfsImage(
            int nx,
            int ny,
            double scale,
            double sigma_arcsec,
            double klim,
            const py::array_t<double>& psf_array,
            bool use_estimate=true,
            int npix_overlap=0,
            int bound=0
        );

        py::array_t<double>
        smooth_image(
            const py::array_t<double>& gal_array,
            bool do_rotate,
            int x,
            int y
        );

        py::array_t<double>
        smooth_image(
            const py::array_t<double>& gal_array,
            const std::optional<py::array_t<double>>& noise_array,
            int xcen,
            int ycen
        );

        void
        find_peaks(
            std::vector<std::tuple<int, int, bool>>& peaks,
            const py::array_t<double>& gal_conv,
            double fthres,
            double pthres,
            double std_m00,
            double std_v,
            int xcen,
            int ycen
        );

        py::array_t<FpfsPeaks>
        detect_source(
            py::array_t<double>& gal_array,
            double fthres,
            double pthres,
            double std_m00,
            double std_v,
            const std::optional<py::array_t<double>>& noise_array=std::nullopt,
            const std::optional<py::array_t<int16_t>>& mask_array=std::nullopt
        );

        py::array_t<double>
        measure_source(
            const py::array_t<double>& gal_array,
            const py::array_t<std::complex<double>>& filter_image,
            const py::array_t<double>& psf_array,
            const std::optional<py::array_t<FpfsPeaks>>& det=std::nullopt,
            bool do_rotate=false
        );

        py::array_t<double>
        measure_source(
            const py::array_t<double>& gal_array,
            const py::array_t<std::complex<double>>& filter_image,
            const BasePsf& psf_obj,
            const std::optional<py::array_t<FpfsPeaks>>& det=std::nullopt,
            bool do_rotate=false
        );

        FpfsImage(FpfsImage&& other) noexcept = default;
        FpfsImage& operator=(FpfsImage&& other) noexcept = default;

        ~FpfsImage();
    };

    void pybindFpfsImage(py::module_& fpfs);
}

#endif // ANACAL_FPFS_IMG_H
