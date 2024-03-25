#ifndef ANACAL_FPFS_H
#define ANACAL_FPFS_H

#include "image.h"

namespace anacal {
    class Fpfs {
    private:
        // Preventing copy (implement these if you need copy semantics)
        Fpfs(const Fpfs&) = delete;
        Fpfs& operator=(const Fpfs&) = delete;
    public:
        double scale = 1.0;
        double sigma_arcsec;
        int nord = 4;
        int det_nrot = 4;
        double klim;
        double sigma_f;

        Fpfs(
            double scale,
            double sigma_arcsec,
            int nord,
            int det_nrot,
            double klim
        );

        py::array_t<double>
        smooth_image(
            const py::array_t<double>& gal_array,
            const py::array_t<double>& psf_array,
            const py::array_t<double>& noise_array
        ) const;

        Fpfs(Fpfs&& other) noexcept = default;
        Fpfs& operator=(Fpfs&& other) noexcept = default;

        ~Fpfs();
    };

    void pyExportFpfs(py::module& m);
}

#endif // FPFS_H
