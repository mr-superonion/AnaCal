#include "anacal.h"

namespace anacal {
namespace ngmix {
void
pyExportNgmix(py::module_& m) {
    py::module_ ngmix = m.def_submodule(
        "ngmix", "submodule for ngmix shape, flux estimation"
    );

    py::class_<modelPrior>(ngmix, "modelPrior")
        .def(py::init<>())
        .def_readonly("w_F", &modelPrior::w_F)
        .def_readonly("w_a", &modelPrior::w_a)
        .def_readonly("w_t", &modelPrior::w_t)
        .def_readonly("w_x", &modelPrior::w_x)
        .def("set_sigma_F", &modelPrior::set_sigma_F,
            "set the Gaussian prior on Flux",
            py::arg("sigma_F")
        )
        .def("set_sigma_t", &modelPrior::set_sigma_t,
            "set the Gaussian prior on log radius",
            py::arg("sigma_t")
        )
        .def("set_sigma_a", &modelPrior::set_sigma_a,
            "set the Gaussian prior on ellipticity",
            py::arg("sigma_a")
        )
        .def("set_sigma_x", &modelPrior::set_sigma_x,
            "set the Gaussian prior on position",
            py::arg("sigma_x")
        );

    py::class_<modelKernelB>(ngmix, "modelKernelB")
        .def(py::init<>())
        .def_readonly("f", &modelKernelB::f);

    py::class_<modelKernelD>(ngmix, "modelKernelD")
        .def(py::init<>())
        .def_readonly("f", &modelKernelD::f)
        .def_readonly("f_a1", &modelKernelD::f_a1)
        .def_readonly("f_a2", &modelKernelD::f_a2);

    py::class_<NgmixGaussian>(ngmix, "NgmixGaussian")
        .def(py::init<bool, bool>(),
            py::arg("force_size")=false,
            py::arg("force_center")=false
        )
        .def_readwrite("F", &NgmixGaussian::F)
        .def_readwrite("t", &NgmixGaussian::t)
        .def_readwrite("a1", &NgmixGaussian::a1)
        .def_readwrite("a2", &NgmixGaussian::a2)
        .def_readwrite("x1", &NgmixGaussian::x1)
        .def_readwrite("x2", &NgmixGaussian::x2)
        .def("prepare_modelD", &NgmixGaussian::prepare_modelD,
            "Prepare the gradient function",
            py::arg("scale"), py::arg("sigma_arcsec")
        )
        .def("get_r2",
             py::overload_cast<
                 double, double, const modelKernelB&
             >(&NgmixGaussian::get_r2, py::const_),
             "Returns the r squared value at x, y using modelKernelB (as a math.qnumber).",
             py::arg("x"), py::arg("y"), py::arg("c")
        )
        .def("get_r2",
             py::overload_cast<
                 double, double, const modelKernelD&
             >(&NgmixGaussian::get_r2, py::const_),
             "Returns the r squared value and its first-order derivatives at x, y using modelKernelD (as a math.lossNumber).",
             py::arg("x"), py::arg("y"), py::arg("c")
        )
        .def("get_model", &NgmixGaussian::get_model,
            "Returns the distorted model value at x, y",
            py::arg("x"), py::arg("y"), py::arg("c")
        )
        .def("get_shape", &NgmixGaussian::get_shape,
            "Returns shape (e1, e2)"
        )
        .def("get_flux_stamp", &NgmixGaussian::get_flux_stamp,
            "Returns the flux on a stamp",
            py::arg("nx"), py::arg("ny"), py::arg("scale"), py::arg("sigma_arcsec")
        )
        .def("get_image_stamp", &NgmixGaussian::get_image_stamp,
            "Returns the image on a stamp",
            py::arg("nx"), py::arg("ny"), py::arg("scale"), py::arg("sigma_arcsec")
        );

    py::class_<GaussFit>(ngmix, "GaussFit")
        .def(
            py::init<
                double, double, int, bool, bool, double
            >(),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("stamp_size")=64,
            py::arg("force_size")=false,
            py::arg("force_center")=false,
            py::arg("fpfs_c0")=1.0
        )
        .def("process_block", &GaussFit::process_block,
            "Run iteration for fitting",
            py::arg("catalog"),
            py::arg("img_array"),
            py::arg("psf_array"),
            py::arg("prior"),
            py::arg("noise_array")=py::none(),
            py::arg("num_epochs")=5,
            py::arg("variance")=1.0,
            py::arg("block")=py::none()
        );
}

} // end of ngmix
} // end of anacal
