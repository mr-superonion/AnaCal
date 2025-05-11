#include "anacal.h"

namespace anacal {
namespace task {

void
pyExportTask(py::module_& m) {
    py::module_ task = m.def_submodule(
        "task", "submodule for task"
    );
    py::class_<TaskAlpha>(task, "TaskAlpha")
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
            py::arg("num_epochs")=15,
            py::arg("force_size")=false,
            py::arg("force_center")=false,
            py::arg("fpfs_c0")=1.0
        )
        .def("process_image",
            py::overload_cast<
                const py::array_t<double>&,
                const py::array_t<double>&,
                double,
                std::vector<geometry::block>&,
                const std::optional<py::array_t<table::galRow>>&,
                const std::optional<py::array_t<double>>&,
                const std::optional<py::array_t<int16_t>>&
            >(&TaskAlpha::process_image),
            "process image with PSF array",
            py::arg("img_array"),
            py::arg("psf_array"),
            py::arg("variance"),
            py::arg("block_list"),
            py::arg("detection")=py::none(),
            py::arg("noise_array")=py::none(),
            py::arg("mask_array")=py::none()
        )
        .def("process_image",
            py::overload_cast<
                const py::array_t<double>&,
                const psf::BasePsf&,
                double,
                std::vector<geometry::block>&,
                const std::optional<py::array_t<table::galRow>>&,
                const std::optional<py::array_t<double>>&,
                const std::optional<py::array_t<int16_t>>&
            >(&TaskAlpha::process_image),
            "process image with PSF object",
            py::arg("img_array"),
            py::arg("psf_obj"),
            py::arg("variance"),
            py::arg("block_list"),
            py::arg("detection")=py::none(),
            py::arg("noise_array")=py::none(),
            py::arg("mask_array")=py::none()
        );
    task.def(
        "get_smoothed_variance", &get_smoothed_variance,
        "get noise variance for smoothed image",
        py::arg("scale"),
        py::arg("sigma_arcsec"),
        py::arg("psf_array"),
        py::arg("variance")
    );
}

} // end of task
} // end of anacal
