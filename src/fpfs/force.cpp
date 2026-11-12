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

    // AfpfsCatRow: the FpfsCatRow fields, then the per-source kernel
    // size of the adaptive measurement and its shear response.
    fpfs.attr("adaptive_catalog_columns") = py::make_tuple(
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
        "flux_err", "s2n", "ds2n_dg1", "ds2n_dg2",
        "sigma", "dsigma_dg1", "dsigma_dg2"
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
            "positions.  psf_array is one (ny, nx) stamp (per-source "
            "PSFs come through psf_model); filter_image holds 9 modes "
            "(through m44: 2nd-order estimator, q and its responses "
            "NaN) or 12 modes (through m64: 4th-order estimator too); "
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
            py::arg("noise2_array")=py::none(),
            py::arg("n_mask_base")=py::none(),
            py::arg("n_mask_base_max")=py::none(),
            py::arg("psf_model")=py::none(),
            py::arg("psf_offset_x")=0.0,
            py::arg("psf_offset_y")=0.0,
            py::arg("renoise")=py::none()
        )
        .def("process_image_adaptive", &ForceTask::process_image_adaptive,
            "Adaptive-kernel ('afpfs') measurement: every source uses "
            "its own kernel size sigma (with shear response "
            "dsigma_dg1/2), interpolated on filter_table "
            "(n_int, nky, nkx, 2 ncol) over the node sizes sigma_grid "
            "(n_int + 1); std_m00_grid is the mode-noise scalar at the "
            "nodes.  Rows follow adaptive_catalog_columns.",
            py::arg("gal_array"),
            py::arg("psf_array"),
            py::arg("filter_table"),
            py::arg("sigma_grid"),
            py::arg("std_m00_grid"),
            py::arg("sigma"),
            py::arg("dsigma_dg1"),
            py::arg("dsigma_dg2"),
            py::arg("det"),
            py::arg("out_dtype"),
            py::arg("noise_array")=py::none(),
            py::arg("noise2_array")=py::none(),
            py::arg("n_mask_base")=py::none(),
            py::arg("n_mask_base_max")=py::none(),
            py::arg("psf_model")=py::none(),
            py::arg("psf_offset_x")=0.0,
            py::arg("psf_offset_y")=0.0,
            py::arg("renoise")=py::none()
        );
    // n_mask_base sentinel for PSF-invalid sources (always skipped).
    fpfs.attr("PSF_INVALID_MASK_VALUE") =
        psfmodel::psf_invalid_mask_value;
}

    } // namespace fpfs
}
