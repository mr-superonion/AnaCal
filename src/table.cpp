#include "anacal.h"

namespace anacal {
namespace table {

void
pyExportTable(py::module_& m) {
    PYBIND11_NUMPY_DTYPE(galRow,
        ra, dec,
        flux, dflux_dg1, dflux_dg2,
        t, dt_dg1, dt_dg2,
        a1, da1_dg1, da1_dg2,
        a2, da2_dg1, da2_dg2,
        e1, de1_dg1, de1_dg2,
        e2, de2_dg1, de2_dg2,
        x1, dx1_dg1, dx1_dg2,
        x2, dx2_dg1, dx2_dg2,
        fluxap2, dfluxap2_dg1, dfluxap2_dg2,
        wdet, dwdet_dg1, dwdet_dg2,
        mask_value, is_peak, is_primary,
        fpfs_e1, fpfs_de1_dg1, fpfs_de1_dg2,
        fpfs_e2, fpfs_de2_dg1, fpfs_de2_dg2,
        fpfs_m0, fpfs_dm0_dg1, fpfs_dm0_dg2,
        fpfs_m2, fpfs_dm2_dg1, fpfs_dm2_dg2,
        peakv, dpeakv_dg1, dpeakv_dg2, bkg, dbkg_dg1, dbkg_dg2
    );
    py::module_ table = m.def_submodule(
        "table", "submodule for table"
    );
    py::class_<galRow>(table, "galRow", py::module_local(false))
        .def(py::init<>())
        .def_readwrite("ra",  &galRow::ra)
        .def_readwrite("dec", &galRow::dec)
        .def_readwrite("flux", &galRow::flux);

    py::class_<galNumber>(table, "galNumber")
        .def(py::init<>())
        .def(py::init<
            ngmix::NgmixGaussian, math::qnumber, math::qnumber,
            int, bool, math::lossNumber
            >(),
            py::arg("model"), py::arg("fluxap2"),
            py::arg("wdet"), py::arg("mask_value"), py::arg("is_peak"),
            py::arg("loss")
        )
        .def_readwrite("model", &galNumber::model)
        .def_readonly("fluxap2", &galNumber::fluxap2)
        .def_readonly("wdet", &galNumber::wdet)
        .def_readonly("mask_value", &galNumber::mask_value)
        .def_readonly("is_peak", &galNumber::is_peak)
        .def_readonly("loss", &galNumber::loss)
        .def_readonly("fpfs_e1", &galNumber::fpfs_e1)
        .def_readonly("fpfs_e2", &galNumber::fpfs_e2)
        .def_readonly("fpfs_m0", &galNumber::fpfs_m0)
        .def_readonly("fpfs_m2", &galNumber::fpfs_m2)
        .def("to_row", &galNumber::to_row)
        .def("from_row", &galNumber::from_row);

    table.def(
        "objlist_to_array", &objlist_to_array,
        "return structured array for catalog",
        py::arg("catalog")
    );
}

} // end of table
} // end of anacal
