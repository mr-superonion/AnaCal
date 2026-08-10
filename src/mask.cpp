#include "anacal.h"

namespace anacal {
namespace mask {
    void
    pyExportMask(py::module& m) {
        PYBIND11_NUMPY_DTYPE(BrightStar, x, y, r);
        PYBIND11_NUMPY_DTYPE(
            Position,
            y, x
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
            "convolve_mask", &convolve_mask,
            "Smooths the mask image with a kernel",
            py::arg("mask_array"),
            py::arg("kernel")
        );
        mask.def(
            "convolve_mask_gauss", &convolve_mask_gauss,
            "Smooths the mask image with a Gaussian kernel",
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale")
        );
        mask.def(
            "add_pixel_mask_column",
            // The C++ function updates the catalog in place, but pybind11
            // hands it a COPY of the Python list, so the update would be
            // lost.  Return the updated catalog instead.
            [](
                std::vector<table::galNumber> catalog,
                const py::array_t<int16_t>& mask_array,
                double sigma,
                double scale
            ) {
                add_pixel_mask_column(catalog, mask_array, sigma, scale);
                return catalog;
            },
            "Return the detection catalog with the pixel mask value updated",
            py::arg("catalog"),
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale")
        );
        mask.def(
            "add_pixel_mask_column",
            // Overload for a structured galRow ARRAY (external detection
            // catalogs): same stamping, array in / array out.  Positions
            // are read from the model centre (x1/x2), which must be in
            // the mask's pixel frame.
            [](
                const py::array_t<table::galRow>& detection,
                const py::array_t<int16_t>& mask_array,
                double sigma,
                double scale
            ) {
                std::vector<table::galNumber> cat =
                    table::array_to_objlist(detection);
                add_pixel_mask_column(cat, mask_array, sigma, scale);
                return table::objlist_to_array(cat);
            },
            "Return the detection ARRAY with the pixel mask value "
            "updated (same stamping as the catalog overload)",
            py::arg("detection"),
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale")
        );
    }
} // end of mask
} // end of anacal
