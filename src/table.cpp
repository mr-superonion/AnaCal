#include "anacal.h"

namespace anacal {
namespace table {

void
pyExportTable(py::module_& m) {
    py::module_ table = m.def_submodule(
        "table", "submodule for table"
    );

    py::class_<galNumber>(table, "galNumber")
        .def(py::init<>())
        .def(py::init<
            ngmix::modelNumber, math::qnumber, int, ngmix::lossNumber
            >(),
            py::arg("params"), py::arg("wedt"), py::arg("mask_value"),
            py::arg("loss")
        )
        .def_readwrite("params", &galNumber::params)
        .def_readwrite("wdet", &galNumber::wdet)
        .def_readwrite("mask_value", &galNumber::mask_value)
        .def_readwrite("loss", &galNumber::loss);

    py::class_<block>(table, "block")
        .def(py::init<>())
        .def(py::init<
            int, int, int, int, int,
            int, int, int, int, int
            >(),
            py::arg("xcen"), py::arg("ycen"), py::arg("xmin"), py::arg("ymin"),
            py::arg("xmax"), py::arg("ymax"), py::arg("xmin_in"),
            py::arg("ymin_in"), py::arg("xmax_in"), py::arg("ymax_in")
        )
        .def_readwrite("xcen", &block::xcen)
        .def_readwrite("ycen", &block::ycen)
        .def_readwrite("xmin", &block::xmin)
        .def_readwrite("ymin", &block::ymin)
        .def_readwrite("xmax", &block::xmax)
        .def_readwrite("ymax", &block::ymax)
        .def_readwrite("xmin_in", &block::xmin_in)
        .def_readwrite("ymin_in", &block::ymin_in)
        .def_readwrite("xmax_in", &block::xmax_in)
        .def_readwrite("ymax_in", &block::ymax_in);

    table.def(
        "get_block_center_list", &get_block_center_list,
        "get the centers of the blocks",
        py::arg("img_nx"),
        py::arg("img_ny"),
        py::arg("block_nx"),
        py::arg("block_ny"),
        py::arg("block_overlap")
    );
}

} // end of table
} // end of anacal
