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
            ngmix::NgmixGaussian, math::tnumber,
            int, bool, math::lossNumber
            >(),
            py::arg("model"), py::arg("wdet"), py::arg("mask_value"),
            py::arg("is_peak"), py::arg("loss")
        )
        .def_readwrite("model", &galNumber::model)
        .def_readwrite("wdet", &galNumber::wdet)
        .def_readwrite("mask_value", &galNumber::mask_value)
        .def_readwrite("is_peak", &galNumber::is_peak)
        .def_readwrite("loss", &galNumber::loss);
}

} // end of table
} // end of anacal
