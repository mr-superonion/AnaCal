#include "anacal.h"

namespace anacal {
namespace table {

void
pyExportTable(py::module_& m) {
    PYBIND11_NUMPY_DTYPE(galRow,
        flux, dflux_dg1, dflux_dg2, rho, drho_dg1, drho_dg2, e1, de1_dg1,
        de1_dg2, e2, de2_dg1, de2_dg2, x, dx_dg1, dx_dg2, y, dy_dg1, dy_dg2,
        wdet, dwdet_dg1, dwdet_dg2, mask_value, is_peak
    );
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
        .def_readwrite("loss", &galNumber::loss)
        .def("to_row", &galNumber::to_row);

    table.def(
        "objlist_to_array", &objlist_to_array,
        "return structured array for catalog",
        py::arg("catalog")
    );
}

} // end of table
} // end of anacal
