#include "anacal.h"

namespace anacal {

void
pyExportFpfsBase(py::module_& fpfs) {
    fpfs.def(
        "gauss_kernel_rfft",
        &gauss_kernel_rfft,
        py::arg("ny"),
        py::arg("nx"),
        py::arg("sigma"),
        py::arg("kmax"),
        py::arg("return_grid") = false
    );
    fpfs.def(
        "shapelets2d_func",
        &shapelets2d_func,
        py::arg("npix"),
        py::arg("norder"),
        py::arg("sigma"),
        py::arg("kmax")
    );
    fpfs.def(
        "shapelets2d",
        &shapelets2d,
        py::arg("norder"),
        py::arg("npix"),
        py::arg("sigma"),
        py::arg("kmax")
    );
    fpfs.def(
        "detlets2d",
        &detlets2d,
        py::arg("npix"),
        py::arg("sigma"),
        py::arg("kmax")
    );
    fpfs.def(
        "get_kmax",
        &get_kmax,
        py::arg("psf_pow"),
        py::arg("sigma"),
        py::arg("kmax_thres") = 1e-20
    );
    fpfs.def(
        "m00_to_flux",
        py::overload_cast<double, double, double>(
            &m00_to_flux
        ),
        py::arg("m00"),
        py::arg("sigma_arcsec"),
        py::arg("pixel_scale")
    );
    fpfs.def(
        "m00_to_flux",
        py::overload_cast<const py::array_t<double>&, double, double>(
            &m00_to_flux
        ),
        py::arg("m00"),
        py::arg("sigma_arcsec"),
        py::arg("pixel_scale")
    );
}

} // namespace anacal
