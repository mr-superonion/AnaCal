#include "anacal.h"

namespace anacal {
namespace geometry {

void
pyExportGeometry(py::module_& m) {
    py::module_ geometry = m.def_submodule(
        "geometry", "submodule for geometry"
    );

    py::class_<block>(geometry, "block")
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
        .def_readwrite("scale", &block::scale)
        .def_readwrite("xcen", &block::xcen)
        .def_readwrite("ycen", &block::ycen)
        .def_readwrite("xmin", &block::xmin)
        .def_readwrite("ymin", &block::ymin)
        .def_readwrite("xmax", &block::xmax)
        .def_readwrite("ymax", &block::ymax)
        .def_readwrite("xmin_in", &block::xmin_in)
        .def_readwrite("ymin_in", &block::ymin_in)
        .def_readwrite("xmax_in", &block::xmax_in)
        .def_readwrite("ymax_in", &block::ymax_in)
        .def_readwrite("scale", &block::scale)
        .def_readwrite("index", &block::index);

    geometry.def(
        "get_block_list", &get_block_list,
        "get a list of blocks",
        py::arg("img_nx"),
        py::arg("img_ny"),
        py::arg("block_nx"),
        py::arg("block_ny"),
        py::arg("block_overlap"),
        py::arg("scale")
    );
}

} // end of geometry
} // end of anacal
