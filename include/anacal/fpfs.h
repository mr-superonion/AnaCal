#ifndef ANACAL_FPFS_H
#define ANACAL_FPFS_H

#include "image.h"

namespace anacal {
    class FpfsDetect {
    private:
        // Preventing copy (implement these if you need copy semantics)
        FpfsDetect(const FpfsDetect&) = delete;
        FpfsDetect& operator=(const FpfsDetect&) = delete;
    public:
        double scale = 1.0;
        double sigma_arcsec;
        int det_nrot = 4;
        double klim;
        double sigma_f;

        FpfsDetect(
            double scale,
            double sigma_arcsec,
            int det_nrot,
            double klim
        );

        py::array_t<double>
        smooth_image(
            const py::array_t<double>& gal_array,
            const py::array_t<double>& psf_array,
            const py::array_t<double>& noise_array
        ) const;

        FpfsDetect(FpfsDetect&& other) noexcept = default;
        FpfsDetect& operator=(FpfsDetect&& other) noexcept = default;

        ~FpfsDetect();
    };

    void pyExportFpfs(py::module& m);
}

#endif // FPFS_H
