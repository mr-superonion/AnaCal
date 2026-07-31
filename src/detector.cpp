#include "anacal.h"

namespace anacal {
namespace detector {
void
pyExportDetector(py::module_& m) {
    py::module_ detector = m.def_submodule(
        "detector", "submodule for detector"
    );
    detector.def("find_peaks", &find_peaks,
        "Find peaks in a block.\n\n"
        "img_array, psf_array and noise_array are either single images or "
        "(nband, ...) stacks; with a stack, variance takes one value per band "
        "and the bands are averaged with inverse-variance weights after each "
        "band's PSF has been removed.  Pass weights to pin that averaging to "
        "a specific set instead of deriving it.",
        py::arg("img_array"),
        py::arg("psf_array"),
        py::arg("sigma_arcsec"),
        py::arg("snr_min"),
        py::arg("variance"),
        py::arg("omega_f"),
        py::arg("omega_v"),
        py::arg("block"),
        py::arg("noise_array")=py::none(),
        py::arg("image_bound")=0,
        py::arg("weights")=py::none()
    );
    detector.def("band_weights", &band_weights,
        "Normalised inverse-variance weights per band at a smoothing scale",
        py::arg("scale"),
        py::arg("sigma_arcsec"),
        py::arg("psf_array"),
        py::arg("variance")
    );
    detector.def("coadd_smoothed_variance", &coadd_smoothed_variance,
        "Variance of the band-averaged smoothed image for fixed weights",
        py::arg("scale"),
        py::arg("sigma_arcsec"),
        py::arg("psf_array"),
        py::arg("variance"),
        py::arg("weights")
    );
}

} // end of detector
} // end of anacal
