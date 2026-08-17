#include "anacal.h"


namespace anacal {
    namespace fpfs {

void
pyExportFpfsForce(py::module_& fpfs) {
    // Canonical FpfsCatRow field order.  Python builds the FINAL output
    // dtype from this list (kernel token + band prefix applied to the
    // names) and hands it to ForceTask.process_image, so the catalog is
    // born with the right column names -- no renaming ever happens.
    fpfs.attr("catalog_columns") = py::make_tuple(
        "e1", "de1_dg1", "de1_dg2",
        "e2", "de2_dg1", "de2_dg2",
        "q1", "dq1_dg1", "dq1_dg2",
        "q2", "dq2_dg1", "dq2_dg2",
        "m00", "dm00_dg1", "dm00_dg2",
        "m20", "dm20_dg1", "dm20_dg2",
        "m22c", "dm22c_dg1", "dm22c_dg2",
        "m22s", "dm22s_dg1", "dm22s_dg2",
        "m42c", "dm42c_dg1", "dm42c_dg2",
        "m42s", "dm42s_dg1", "dm42s_dg2",
        "m00_err",
        "flux", "dflux_dg1", "dflux_dg2",
        "flux_err", "s2n", "ds2n_dg1", "ds2n_dg2"
    );

    py::class_<ForceTask>(fpfs, "ForceTask")
        .def(py::init<int, double, double, double, double>(),
            "Forced FPFS measurement for one shapelet kernel; the whole "
            "per-source loop (modes, noise subtraction, shear responses, "
            "flux family) runs with the GIL released and returns the "
            "finished catalog, like task.Task.process_image.",
            py::arg("npix"),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("klim"),
            py::arg("c0")
        )
        .def("process_image", &ForceTask::process_image,
            "Measure the finished per-kernel catalog at the given "
            "positions.  psf_array is one (ny, nx) stamp or an "
            "(nsrc, ny, nx) stack of pre-drawn per-source stamps; "
            "std_m00 is the mode-noise scalar from "
            "FpfsTask.prepare_covariance; out_dtype carries the final "
            "column names (see catalog_columns for the field order).",
            py::arg("gal_array"),
            py::arg("psf_array"),
            py::arg("filter_image"),
            py::arg("det"),
            py::arg("std_m00"),
            py::arg("out_dtype"),
            py::arg("noise_array")=py::none(),
            py::arg("n_mask_base")=py::none(),
            py::arg("n_mask_base_max")=py::none(),
            py::arg("psf_model")=py::none(),
            py::arg("psf_offset_x")=0.0,
            py::arg("psf_offset_y")=0.0
        );
    // n_mask_base sentinel for PSF-invalid sources (always skipped).
    fpfs.attr("PSF_INVALID_MASK_VALUE") =
        psfmodel::psf_invalid_mask_value;
}

    } // namespace fpfs
}
