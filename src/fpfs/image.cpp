#include "anacal.h"

namespace anacal {
    namespace fpfs {

void
pyExportFpfsImage(py::module_& fpfs) {
    py::class_<FpfsImage>(fpfs, "FpfsImage")
        .def(py::init<
                int, int, double, double, double, const py::array_t<double>&,
                bool
            >(),
            "Initialize the FpfsImage object using an ndarray",
            py::arg("nx"), py::arg("ny"),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("klim"),
            py::arg("psf_array"),
            py::arg("use_estimate")=true
        )
        .def("measure_source",
            &FpfsImage::measure_source,
            "measure source properties using filter at the position of det",
            py::arg("gal_array"),
            py::arg("filter_image"),
            py::arg("psf_array"),
            py::arg("det")=py::none(),
            py::arg("do_rotate")=false
        )
        .def("measure_source_at",
            &FpfsImage::measure_source_at,
            "Measure source properties using the filter at a single position",
            py::arg("gal_array"),
            py::arg("filter_image"),
            py::arg("psf_array"),
            py::arg("y"),
            py::arg("x"),
            py::arg("do_rotate")=false
        );
}

    } // namespace fpfs
} // namespace anacal
