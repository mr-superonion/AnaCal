#include "anacal.h"

namespace anacal {
namespace geometry {

void
pyExportGeometry(py::module_& m) {
    py::module_ geometry = m.def_submodule(
        "geometry", "submodule for geometry"
    );

    py::class_<cell>(geometry, "cell")
        .def(py::init<>())
        .def(py::init<
            int, int, int, int, int,
            int, int, int, int, int,
            double, int
            >(),
            py::arg("xcen"), py::arg("ycen"), py::arg("xmin"), py::arg("ymin"),
            py::arg("xmax"), py::arg("ymax"), py::arg("xmin_in"),
            py::arg("ymin_in"), py::arg("xmax_in"), py::arg("ymax_in"),
            py::arg("scale"), py::arg("index")
        )
        .def_readwrite("scale", &cell::scale)
        .def_readwrite("xcen", &cell::xcen)
        .def_readwrite("ycen", &cell::ycen)
        .def_readwrite("xmin", &cell::xmin)
        .def_readwrite("ymin", &cell::ymin)
        .def_readwrite("xmax", &cell::xmax)
        .def_readwrite("ymax", &cell::ymax)
        .def_readwrite("xmin_in", &cell::xmin_in)
        .def_readwrite("ymin_in", &cell::ymin_in)
        .def_readwrite("xmax_in", &cell::xmax_in)
        .def_readwrite("ymax_in", &cell::ymax_in)
        .def_readwrite("nx", &cell::nx)
        .def_readwrite("ny", &cell::ny)
        .def_readwrite("xvs", &cell::xvs)
        .def_readwrite("yvs", &cell::yvs)
        .def_readwrite("xmsk", &cell::xmsk)
        .def_readwrite("ymsk", &cell::ymsk)
        .def_readwrite("psf_array", &cell::psf_array)
        .def_readwrite("index", &cell::index);

    geometry.def(
        "get_cell_list", &get_cell_list,
        "get a list of cells",
        py::arg("img_nx"),
        py::arg("img_ny"),
        py::arg("cell_nx"),
        py::arg("cell_ny"),
        py::arg("cell_overlap"),
        py::arg("scale")
    );
}

} // end of geometry
} // end of anacal
