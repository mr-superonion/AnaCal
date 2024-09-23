#ifndef ANACAL_FPFS_H
#define ANACAL_FPFS_H

#include "image.h"
#include "psf.h"
#include "math.h"
#include "mask.h"
#include "stdafx.h"

namespace anacal {
    inline constexpr double fpfs_cut_sigma_ratio = 1.6;
    inline constexpr double fpfs_det_sigma2 = 0.04;
    inline constexpr double fpfs_pnr = 0.8;

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


    inline FpfsShapeletsResponse calculate_shapelets_dg(
        const FpfsShapelets& x
    ) {
        double m00_g1 = -std::sqrt(2.0) * x.m22c;
        double m00_g2 = -std::sqrt(2.0) * x.m22s;
        double m20_g1 = -std::sqrt(6.0) * x.m42c;
        double m20_g2 = -std::sqrt(6.0) * x.m42s;

        double m22c_g1 = (
            1.0 / std::sqrt(2.0)
        ) * (x.m00 - x.m40) - std::sqrt(3.0) * x.m44c;
        double m22s_g2 = (
            1.0 / std::sqrt(2.0)
        ) * (x.m00 - x.m40) + std::sqrt(3.0) * x.m44c;

        // Off-diagonal terms
        // double m22c_g2 = -std::sqrt(3.0) * x.m44s;
        // double m22s_g1 = -std::sqrt(3.0) * x.m44s;

        double m42c_g1 = (
            std::sqrt(6.0) / 2.0
        ) * (x.m20 - x.m60) - std::sqrt(5.0) * x.m64c;
        double m42s_g2 = (
            std::sqrt(6.0) / 2.0
        ) * (x.m20 - x.m60) + std::sqrt(5.0) * x.m64c;

        // Off-diagonal terms
        // double m42c_g2 = -std::sqrt(5.0) * x.m64s;
        // double m42s_g1 = -std::sqrt(5.0) * x.m64s;

        return FpfsShapeletsResponse{
            m00_g1, m00_g2, m20_g1,
            m20_g2, m22c_g1, m22s_g2,
            m42c_g1, m42s_g2
        };
    };

    inline FpfsShapeletsResponse calculate_shapelets_dg(
        const FpfsDetect& x
    ) {
        double m00_g1 = -std::sqrt(2.0) * x.m22c;
        double m00_g2 = -std::sqrt(2.0) * x.m22s;
        double m20_g1 = -std::sqrt(6.0) * x.m42c;
        double m20_g2 = -std::sqrt(6.0) * x.m42s;

        double m22c_g1 = (
            1.0 / std::sqrt(2.0)
        ) * (x.m00 - x.m40) - std::sqrt(3.0) * x.m44c;
        double m22s_g2 = (
            1.0 / std::sqrt(2.0)
        ) * (x.m00 - x.m40) + std::sqrt(3.0) * x.m44c;

        // Off-diagonal terms
        // double m22c_g2 = -std::sqrt(3.0) * x.m44s;
        // double m22s_g1 = -std::sqrt(3.0) * x.m44s;

        double m42c_g1 = (
            std::sqrt(6.0) / 2.0
        ) * (x.m20 - x.m60) - std::sqrt(5.0) * x.m64c;
        double m42s_g2 = (
            std::sqrt(6.0) / 2.0
        ) * (x.m20 - x.m60) + std::sqrt(5.0) * x.m64c;

        // Off-diagonal terms
        // double m42c_g2 = -std::sqrt(5.0) * x.m64s;
        // double m42s_g1 = -std::sqrt(5.0) * x.m64s;

        return FpfsShapeletsResponse{
            m00_g1, m00_g2, m20_g1,
            m20_g2, m22c_g1, m22s_g2,
            m42c_g1, m42s_g2
        };
    };

    inline FpfsShapeCatalog calculate_fpfs_ell(
        const FpfsShapelets& x, const FpfsShapeletsResponse& x_dg, double C0
    ) {
        // Denominator
        double _denom = x.m00 + C0;

        // Compute ellipticity 1
        double e1 = x.m22c / _denom;
        double e1_g1 = x_dg.m22c_g1 / _denom
            - (x_dg.m00_g1 * x.m22c) / (_denom * _denom);

        // Compute ellipticity 2
        double e2 = x.m22s / _denom;
        double e2_g2 = x_dg.m22s_g2 / _denom
            - (x_dg.m00_g2 * x.m22s) / (_denom * _denom);

        // Compute ellipticity 1 (4th order)
        double q1 = x.m42c / _denom;
        double q1_g1 = x_dg.m42c_g1 / _denom
            - (x_dg.m00_g1 * x.m42c) / (_denom * _denom);

        // Compute ellipticity 2 (4th order)
        double q2 = x.m42s / _denom;
        double q2_g2 = x_dg.m42s_g2 / _denom
            - (x_dg.m00_g2 * x.m42s) / (_denom * _denom);

        // Return the result as FpfsShapeCatalog
        return FpfsShapeCatalog{e1, e1_g1, e2_g2, q1, q1_g1, q2_g2};
    }


    void pyExportFpfs(py::module& m);
}

#endif // ANACAL_FPFS_H
