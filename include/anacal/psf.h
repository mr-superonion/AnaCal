#ifndef ANACAL_PSF_H
#define ANACAL_PSF_H

#include "stdafx.h"

namespace anacal {
    class BasePsf {
    public:
        BasePsf();

        BasePsf(BasePsf&& other) noexcept = default;
        BasePsf& operator=(BasePsf&& other) noexcept = default;

        virtual ~BasePsf() = default;

    };


    /// Gaussian Function
    class GridPsf : public BasePsf {
    private:
        double x0, y0;
        double dx, dy;
        int nx, ny;
        int ngrid;
        py::array_t<double> model_array;
        py::slice slice;
    public:
        GridPsf(
            double x0,
            double y0,
            double dx,
            double dy,
            py::array_t<double> model_array
        );
        py::array_t<double>
        draw(double x, double y) const;
    };

    void pyExportPsf(py::module& m);
}
#endif // ANACAL_PSF_H
