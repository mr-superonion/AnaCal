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
            double, double, double, double,
            const std::optional<ngmix::modelPrior>,
            int, int, int,
            bool, bool, double
            >(),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("snr_peak_min"),
            py::arg("omega_f"),
            py::arg("v_min"),
            py::arg("omega_v"),
            py::arg("p_min"),
            py::arg("omega_p"),
            py::arg("prior")=py::none(),
            py::arg("stamp_size")=32,
            py::arg("image_bound")=0,
            py::arg("num_epochs")=3,
            py::arg("force_size")=false,
            py::arg("force_center")=false,
            py::arg("fpfs_c0")=1.0
        )
        .def("process_image", &Task::process_image,
            "process image with PSF array",
            py::arg("img_array"),
            py::arg("psf_array"),
            py::arg("variance"),
            py::arg("block_list"),
            py::arg("detection")=py::none(),
            py::arg("noise_array")=py::none(),
            py::arg("mask_array")=py::none(),
            py::arg("a_ini")=0.2
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
