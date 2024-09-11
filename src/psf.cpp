#include "anacal.h"


namespace anacal {

    BasePsf::BasePsf(): crun(true) {
        // Constructor implementation. Can be empty if nothing to initialize.
    }

    py::array_t<double>
    BasePsf::draw(double, double) const {
        py::array_t<double> result = py::array_t<double>({1, 1});
        *result.mutable_data(0, 0) = 0.0;
        return result;
    }

    GridPsf::GridPsf(
        double x0,
        double y0,
        double dx,
        double dy,
        py::array_t<double> model_array
    ) : x0(x0), y0(y0), dx(dx), dy(dy), model_array(model_array), crun(true) {
        int ndim = model_array.ndim();
        if (ndim != 4) {
            throw std::runtime_error("PSF model array has wrong shape");
        }
        this->ny =  model_array.shape()[0];
        this->nx =  model_array.shape()[1];
        ngrid = model_array.shape()[2];
        int ngrid2 = model_array.shape()[3];
        if (ngrid != ngrid2) {
            throw std::runtime_error("the PSF model is not in square stamp");
        }
        this->slice = py::slice(0, ngrid, 1);
    }

    py::array_t<double>
    GridPsf::draw(double x, double y) const {
        int x_grid = static_cast<int>((x - this->x0) / this->dx);
        int y_grid = static_cast<int>((y - this->y0) / this->dy);
        if (x_grid < 0) x_grid = 0;
        if (y_grid < 0) y_grid = 0;
        if (x_grid >= this->nx) x_grid = this->nx - 1;
        if (y_grid >= this->ny) y_grid = this->ny - 1;

        py::array_t<double> view = py::array_t<double>(
            this->model_array[
                py::make_tuple(
                    y_grid, x_grid, this->slice, this->slice
                )
            ]
        );
        auto vinfo = view.request();
        py::array_t<double> result(vinfo.shape);
        std::copy(
            view.data(),
            view.data() + vinfo.size,
            result.mutable_data()
        );
        return result;
    }

    void pyExportPsf(py::module& m) {
        py::module_ model = m.def_submodule("psf", "submodule for models");
        py::class_<BasePsf>(model, "BasePsf")
            .def(py::init<>())
            .def_readonly("crun", &BasePsf::crun);

        py::class_<GridPsf, BasePsf>(model, "GridPsf")
            .def(py::init<
                    double, double, double, double,
                    py::array_t<double>
                >(),
                "GridPsf constructor",
                py::arg("x0"), py::arg("y0"),
                py::arg("dx"), py::arg("dy"),
                py::arg("model_array")
            )
            .def("draw", &GridPsf::draw,
                "draw the PSF model image",
                py::arg("x"), py::arg("y")
            )
            .def_readonly("crun", &GridPsf::crun);

         py::class_<PyPsf, BasePsf>(model, "PyPsf")
            .def(py::init<>())
            .def("draw", &PyPsf::draw)
            .def_readonly("crun", &PyPsf::crun);
    }


}
