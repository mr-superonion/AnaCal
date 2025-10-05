#include "anacal.h"

namespace anacal {

void
pyExportFpfsImage(py::module_& fpfs) {
    py::class_<FpfsImage>(fpfs, "FpfsImage")
        .def(py::init<
                int, int, double, double, double, const py::array_t<double>&,
                bool, int, int
            >(),
            "Initialize the FpfsImage object using an ndarray",
            py::arg("nx"), py::arg("ny"),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("klim"),
            py::arg("psf_array"),
            py::arg("use_estimate")=true,
            py::arg("npix_overlap")=0,
            py::arg("bound")=0
        )
        .def("smooth_image",
            py::overload_cast<
                const py::array_t<double>&,
                int,
                int,
                bool
            >(&FpfsImage::smooth_image),
            "Smooths the image after PSF deconvolution",
            py::arg("gal_array"),
            py::arg("xcen"),
            py::arg("ycen"),
            py::arg("do_rotate")=false
        )
        .def("smooth_image",
            py::overload_cast<
                const py::array_t<double>&,
                int,
                int,
                const std::optional<py::array_t<double>>&
            >(&FpfsImage::smooth_image),
            "Smooths the image after PSF deconvolution",
            py::arg("gal_array"),
            py::arg("xcen"),
            py::arg("ycen"),
            py::arg("noise_array")=py::none()
        )
        .def("detect_source",
            &FpfsImage::detect_source,
            "Detect galaxy candidates from image",
            py::arg("gal_array"),
            py::arg("fthres"),
            py::arg("pthres"),
            py::arg("std_m00"),
            py::arg("v_min"),
            py::arg("omega_v"),
            py::arg("noise_array")=py::none(),
            py::arg("mask_array")=py::none()
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

} // namespace anacal
