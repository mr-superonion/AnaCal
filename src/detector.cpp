#include "anacal.h"

namespace anacal {
namespace detector {
void
pyExportDetector(py::module_& m) {
    py::module_ detector = m.def_submodule(
        "detector", "submodule for detector"
    );
    detector.def("find_peaks", &find_peaks,
        "find peak from block",
        py::arg("img_array"),
        py::arg("psf_array"),
        py::arg("sigma_arcsec"),
        py::arg("snr_min"),
        py::arg("variance"),
        py::arg("omega_f"),
        py::arg("v_min"),
        py::arg("omega_v"),
        py::arg("p_min"),
        py::arg("omega_p"),
        py::arg("block"),
        py::arg("noise_array")=py::none(),
        py::arg("image_bound")=0
    );
}

} // end of detector
} // end of anacal
