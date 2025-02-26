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
            double, double, double, double, double, double, double, double,
            ngmix::modelPrior, int, int, int
            >(),
            py::arg("scale"), py::arg("sigma_arcsec_det"),
            py::arg("sigma_arcsec"), py::arg("snr_peak_min"),
            py::arg("omega_f"), py::arg("v_min"), py::arg("omega_v"),
            py::arg("pthres"), py::arg("prior"), py::arg("stamp_size"),
            py::arg("image_bound"), py::arg("num_epochs")
        )
        .def("process_image", &TaskAlpha::process_image);
}

} // end of task
} // end of anacal
