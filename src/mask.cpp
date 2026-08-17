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
        // The two in-place mask mutators guard the dtype by hand: pybind
        // would otherwise CONVERT a mismatched array into a copy, mutate
        // the copy, and silently discard every write (the same trap
        // mask_galaxy_image documents for its image argument).
        mask.def(
            "add_bright_star_mask", &add_bright_star_mask,
            "Flag bright-star footprints into a mask image.\n\n"
            "MODIFIES mask_array IN PLACE (bitwise-ORs bit 0 into every "
            "pixel inside a star radius) and returns None.  mask_array "
            "must already be uint8: it is written in place, so a "
            "different dtype raises rather than being silently "
            "converted to a copy that the caller never sees.",
            py::arg("mask_array"),
            py::arg("star_array")
        );
        mask.def(
            "mask_galaxy_image", &mask_galaxy_image,
            "Zero the galaxy pixels that bit 0 of the mask flags; bit 1 "
            "(discontinuity) pixels keep their data.\n\n"
            "MODIFIES BOTH ARRAYS IN PLACE and returns None: gal_array "
            "pixels with mask_array bit 0 set are zeroed, and when "
            "star_array is given its footprints are first flagged into "
            "mask_array.  gal_array must already be float32 and "
            "mask_array uint8 -- both are written in place, so any "
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
            "gaussian_average_at_sources", &gaussian_average_at_sources,
            "Gaussian-weighted MEAN of an image at source positions.\n\n"
            "Same kernel as add_mask_fraction_columns -- normalised, "
            "sum(K * image) / sum(K) -- and returned as an array rather "
            "than stamped onto the catalog, so it can carry a per-BAND "
            "quantity such as the number of coadd inputs. Positions are "
            "pixels in the image's frame; sources outside it get 0.",
            py::arg("image"),
            py::arg("x_pixel"),
            py::arg("y_pixel"),
            py::arg("sigma"),
            py::arg("scale")
        );
        mask.def(
            "add_mask_fraction_columns",
            // The C++ function updates the catalog in place, but pybind11
            // hands it a COPY of the Python list, so the update would be
            // lost.  Return the updated catalog instead.
            [](
                std::vector<table::galNumber> catalog,
                const py::array_t<uint8_t>& mask_array,
                double sigma,
                double scale
            ) {
                add_mask_fraction_columns(catalog, mask_array, sigma, scale);
                return catalog;
            },
            "Return the catalog with n_mask_base / n_mask_discontinuity\n"
            "set to the Gaussian-weighted mask fractions in [0, 1]",
            py::arg("catalog"),
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale")
        );
        mask.def(
            "add_mask_fraction_columns",
            // Overload for a structured galRow ARRAY (external detection
            // catalogs): same stamping, array in / array out.  Positions
            // are read from the model centre (x1/x2), which must be in
            // the mask's pixel frame.
            [](
                const py::array_t<table::galRow>& detection,
                const py::array_t<uint8_t>& mask_array,
                double sigma,
                double scale
            ) {
                std::vector<table::galNumber> cat =
                    table::array_to_objlist(detection);
                add_mask_fraction_columns(cat, mask_array, sigma, scale);
                return table::objlist_to_array(cat);
            },
            "Return the detection ARRAY with the mask fractions set "
            "(same stamping as the catalog overload)",
            py::arg("detection"),
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale")
        );
    }
} // end of mask
} // end of anacal
