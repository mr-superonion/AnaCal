#ifndef ANACAL_PSF_H
#define ANACAL_PSF_H

#include "stdafx.h"

namespace anacal {
    class BasePsf {
    public:
        BasePsf();

        BasePsf(BasePsf&& other) noexcept = default;
        BasePsf& operator=(BasePsf&& other) noexcept = default;
        virtual py::array_t<double> draw(double, double) const;
        virtual ~BasePsf() = default;
        bool crun;
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
        bool crun;
    };

    class PyPsf : public BasePsf {
    public:
        PyPsf() {this->crun = false;}
        py::array_t<double> draw(double x, double y) const override {
            PYBIND11_OVERRIDE_PURE(
                py::array_t<double>, // Return type
                PyPsf,         // Parent class
                draw,                // Name of the method in Python
                x, y                 // Arguments
            );
        }
        bool crun;
    };

    void pyExportPsf(py::module& m);
}
#endif // ANACAL_PSF_H
