#include "anacal.h"

namespace anacal {
void
pyExportNgmix(py::module_& m) {
    py::module_ ngmix = m.def_submodule(
        "ngmix", "submodule for ngmix shape, flux estimation"
    );

    py::class_<ngmix::modelNumber>(ngmix, "modelNumber")
    .def(py::init<
        math::qnumber, math::qnumber, math::qnumber, math::qnumber,
        math::qnumber, math::qnumber, math::qnumber, math::qnumber,
        math::qnumber, math::qnumber, math::qnumber, math::qnumber,
        math::qnumber>(),
        py::arg("v"),
        py::arg("v_A"), py::arg("v_rho"), py::arg("v_g1"), py::arg("v_g2"),
        py::arg("v_x"), py::arg("v_y"),
        py::arg("v_AA"), py::arg("v_rhorho"), py::arg("v_g1g1"),
        py::arg("v_g2g2"), py::arg("v_xx"), py::arg("v_yy")
    )
    .def_readwrite("v", &ngmix::modelNumber::v)
    .def_readwrite("v_A", &ngmix::modelNumber::v_A)
    .def_readwrite("v_rho", &ngmix::modelNumber::v_rho)
    .def_readwrite("v_g1", &ngmix::modelNumber::v_g1)
    .def_readwrite("v_g2", &ngmix::modelNumber::v_g2)
    .def_readwrite("v_x", &ngmix::modelNumber::v_x)
    .def_readwrite("v_y", &ngmix::modelNumber::v_y)
    .def_readwrite("v_AA", &ngmix::modelNumber::v_AA)
    .def_readwrite("v_rhorho", &ngmix::modelNumber::v_rhorho)
    .def_readwrite("v_g1g1", &ngmix::modelNumber::v_g1g1)
    .def_readwrite("v_g2g2", &ngmix::modelNumber::v_g2g2)
    .def_readwrite("v_xx", &ngmix::modelNumber::v_xx)
    .def_readwrite("v_yy", &ngmix::modelNumber::v_yy);

    py::class_<ngmix::NgmixModel>(ngmix, "NgmixModel")
        .def(py::init<>())
        .def("set_params", &ngmix::NgmixModel::set_params,
            "Sets the transform parameters",
            py::arg("params")
        )
        .def("get_r2", &ngmix::NgmixModel::get_r2,
            "Returns the r squared value at x, y",
            py::arg("x"), py::arg("y")
        )
        .def("model", &ngmix::NgmixModel::model,
            "Returns the distorted model value at x, y",
            py::arg("x"), py::arg("y")
        )
        .def("loss", &ngmix::NgmixModel::loss,
            "Returns the loss function value and the derivatives wrt params",
            py::arg("image_val"), py::arg("variance_val"),
            py::arg("x"), py::arg("y")
        );

    py::class_<ngmix::NgmixGaussian, ngmix::NgmixModel>(ngmix, "NgmixGaussian")
        .def(py::init<double>(),
            "NgmixGaussian basis function",
            py::arg("sigma")
        );

    py::class_<ngmix::GaussFit>(ngmix, "GaussFit")
        .def(
            py::init< int, int, double, double, double, bool >(),
            py::arg("nx"),
            py::arg("ny"),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("klim"),
            py::arg("use_estimate")=true
        )
        .def("run", &ngmix::GaussFit::run,
            "Run iteration for fitting",
            py::arg("params0"), py::arg("xcen"), py::arg("ycen"),
            py::arg("img_array"), py::arg("psf_array"),
            py::arg("noise_array")=py::none(), py::arg("num_epochs")=5,
            py::arg("learning_rate")=1.0
        );
}
}
