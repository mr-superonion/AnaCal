#include "anacal.h"

namespace anacal {
namespace task {

void
pyExportTask(py::module_& m) {

    py::module_ task = m.def_submodule(
        "task", "submodule for task"
    );
    py::class_<Task>(task, "Task")
        .def(py::init<
            double, double, double, double,
            double,
            const std::optional<ngmix::modelPrior>,
            int, int, int,
            bool, bool, double,
            double
            >(),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("snr_peak_min"),
            py::arg("omega_f"),
            py::arg("omega_v"),
            py::arg("prior")=py::none(),
            py::arg("stamp_size")=64,
            py::arg("image_bound")=0,
            py::arg("num_epochs")=3,
            py::arg("force_size")=false,
            py::arg("force_center")=false,
            py::arg("fpfs_c0")=1.0,
            py::arg("mag_zero")=Task::THRESHOLD_REF_MAG_ZERO
        )
        .def("process_image", &Task::process_image,
            "Detect and measure sources.\n\n"
            "img_array, psf_array and noise_array are either single images -- "
            "(ny, nx) and (npsf, npsf) -- or (nband, ...) stacks.  With a "
            "stack, variance takes one value per band and the bands are "
            "combined, after each band's PSF has been removed, into one "
            "inverse-variance weighted detection image.",
            py::arg("img_array"),
            py::arg("psf_array"),
            py::arg("variance"),
            py::arg("cell_list"),
            py::arg("detection")=py::none(),
            py::arg("noise_array")=py::none(),
            py::arg("mask_array")=py::none(),
            py::arg("a_ini")=0.2,
            py::arg("do_measure")=true,
            py::arg("do_fpfs")=true,
            py::arg("mask_value_max")=py::none()
        );
    // Single source of truth for the zeropoint at which the flux-scale
    // thresholds are defined; the Python FPFS path reads it from here rather
    // than duplicating the literal.
    task.attr("THRESHOLD_REF_MAG_ZERO") = Task::THRESHOLD_REF_MAG_ZERO;
    task.def(
        "assign_cell_ids",
        [](const py::array_t<table::galRow>& detection,
           const std::vector<geometry::cell>& cell_list) {
            std::vector<table::galNumber> cat =
                table::array_to_objlist(detection);
            assign_cell_ids(cat, cell_list);
            py::array_t<int> out(static_cast<ssize_t>(cat.size()));
            auto r = out.mutable_unchecked<1>();
            for (ssize_t i = 0;
                 i < static_cast<ssize_t>(cat.size()); ++i) {
                r(i) = cat[i].cell_id;
            }
            return out;
        },
        "Owner cell index for each detection position -- the same rule "
        "process_image applies internally: half-open inner regions "
        "(a source on a shared edge belongs to the right/top cell), "
        "nearest cell for positions outside every inner region.",
        py::arg("detection"),
        py::arg("cell_list")
    );
    task.def(
        "gaussian_flux_variance",
        &gaussian_flux_variance,
        "Compute Gaussian-weighted flux variance for a PSF",
        py::arg("psf_array"),
        py::arg("sigma_kernel"),
        py::arg("sigma_smooth"),
        py::arg("pixel_scale")=1.0,
        py::arg("klim")=std::numeric_limits<double>::infinity(),
        py::arg("noise_corr")=py::none()
    );
}

} // end of task
} // end of anacal
