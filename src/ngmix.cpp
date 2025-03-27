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
        .def_readwrite("mu_F", &modelPrior::mu_F)
        .def_readwrite("mu_t", &modelPrior::mu_t)
        .def_readwrite("w_F", &modelPrior::w_F)
        .def_readwrite("w_e", &modelPrior::w_e)
        .def_readwrite("w_t", &modelPrior::w_t)
        .def_readwrite("w_x", &modelPrior::w_x)
        .def("set_sigma_F", &modelPrior::set_sigma_F,
            "set the Gaussian prior on Flux",
            py::arg("sigma_F")
        )
        .def("set_sigma_t", &modelPrior::set_sigma_t,
            "set the Gaussian prior on log radius",
            py::arg("sigma_t")
        )
        .def("set_sigma_e", &modelPrior::set_sigma_e,
            "set the Gaussian prior on ellipticity",
            py::arg("sigma_e")
        )
        .def("set_sigma_x", &modelPrior::set_sigma_x,
            "set the Gaussian prior on position",
            py::arg("sigma_x")
        );

    py::class_<modelKernel>(ngmix, "modelKernel")
        .def(py::init<>())
        .def_readonly("f", &modelKernel::f)
        .def_readonly("f_t", &modelKernel::f_t)
        .def_readonly("f_e1", &modelKernel::f_e1)
        .def_readonly("f_e2", &modelKernel::f_e2)
        .def_readonly("f_tt", &modelKernel::f_tt)
        .def_readonly("f_e1e1", &modelKernel::f_e1e1)
        .def_readonly("f_e2e2", &modelKernel::f_e2e2);

    py::class_<NgmixGaussian>(ngmix, "NgmixGaussian")
        .def(py::init<bool, bool, bool>(),
            py::arg("force_size")=false,
            py::arg("force_shape")=false,
            py::arg("force_center")=false
        )
        .def_readwrite("F", &NgmixGaussian::F)
        .def_readwrite("t", &NgmixGaussian::t)
        .def_readwrite("e1", &NgmixGaussian::e1)
        .def_readwrite("e2", &NgmixGaussian::e2)
        .def_readwrite("x1", &NgmixGaussian::x1)
        .def_readwrite("x2", &NgmixGaussian::x2)
        .def("prepare_model", &NgmixGaussian::prepare_model,
            "Prepare the gradient function",
            py::arg("scale"), py::arg("sigma_arcsec")
        )
        .def("get_r2", &NgmixGaussian::get_r2,
            "Returns the r squared value at x, y",
            py::arg("x"), py::arg("y"), py::arg("c")
        )
        .def("get_model", &NgmixGaussian::get_model,
            "Returns the distorted model value at x, y",
            py::arg("x"), py::arg("y"), py::arg("c")
        )
        .def("get_loss", &NgmixGaussian::get_loss,
            "Returns the loss function value and the derivatives wrt params",
            py::arg("image_val"), py::arg("variance_val"),
            py::arg("x"), py::arg("y"), py::arg("c")
        )
        .def("get_flux_stamp", &NgmixGaussian::get_flux_stamp,
            "Returns the flux on a stamp",
            py::arg("nx"), py::arg("ny"), py::arg("scale"), py::arg("sigma_arcsec")
        )
        .def("get_image_stamp", &NgmixGaussian::get_image_stamp,
            "Returns the flux on a stamp",
            py::arg("nx"), py::arg("ny"), py::arg("scale"), py::arg("sigma_arcsec")
        );

    py::class_<GaussFit>(ngmix, "GaussFit")
        .def(
            py::init<
                double, double, int, bool, bool, bool, double
            >(),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("stamp_size")=64,
            py::arg("force_size")=false,
            py::arg("force_shape")=false,
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
