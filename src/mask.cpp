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
            "Flag bright-star footprints into a mask image.\n\n"
            "MODIFIES mask_array IN PLACE (bitwise-ORs 1 into every "
            "pixel inside a star radius) and returns None.  mask_array "
            "must already be int16: it is written in place, so a "
            "different dtype raises rather than being silently "
            "converted to a copy that the caller never sees.",
            py::arg("mask_array"),
            py::arg("star_array")
        );
        mask.def(
            "mask_galaxy_image", &mask_galaxy_image,
            "Zero the galaxy pixels that the mask flags.\n\n"
            "MODIFIES BOTH ARRAYS IN PLACE and returns None: gal_array "
            "pixels with mask_array > 0 are set to 0, and when "
            "star_array is given its footprints are first flagged into "
            "mask_array.  gal_array must already be float32 and "
            "mask_array int16 -- both are written in place, so any "
            "other dtype raises rather than being silently converted "
            "to a copy that the caller never sees.",
            py::arg("gal_array"),
            py::arg("mask_array"),
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
