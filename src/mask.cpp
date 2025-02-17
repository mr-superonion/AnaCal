#include "anacal.h"

namespace anacal {
namespace mask {
    void
    pyExportMask(py::module& m) {
        PYBIND11_NUMPY_DTYPE(BrightStar, x, y, r);
        PYBIND11_NUMPY_DTYPE(
            FpfsPeaks,
            y, x,
            is_peak,
            mask_value
        );
        py::module_ mask = m.def_submodule("mask", "submodule for mask");
        mask.def(
            "add_bright_star_mask", &add_bright_star_mask,
            "Update mask image according to bright star catalog",
            py::arg("mask_array"),
            py::arg("star_array")
        );
        mask.def(
            "extend_mask_image", &extend_mask_image,
            "Update mask image with a 2 pixel extension",
            py::arg("mask_array")
        );
        mask.def(
            "mask_galaxy_image", &mask_galaxy_image,
            "Apply mask on galaxy image",
            py::arg("gal_array"),
            py::arg("mask_array"),
            py::arg("do_extend_mask")=true,
            py::arg("star_array")=py::none()
        );
        mask.def(
            "smooth_mask_image", &smooth_mask_image,
            "Smooths the mask image",
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale")
        );
        mask.def(
            "add_pixel_mask_column", &add_pixel_mask_column,
            "Update the detection catalog with the pixel mask value",
            py::arg("det"),
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale")
        );
    }
} // end of mask
} // end of anacal
