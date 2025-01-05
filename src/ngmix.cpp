#include "anacal.h"

namespace anacal {
void
pyExportNgmix(py::module_& m) {
    py::module_ ngmix = m.def_submodule(
        "ngmix", "submodule for ngmix shape, flux estimation"
    );

    py::class_<ngmix::vDeriv>(ngmix, "vDeriv")
    .def(py::init<math::qnumber, math::qnumber, math::qnumber, math::qnumber,
                  math::qnumber, math::qnumber, math::qnumber, math::qnumber,
                  math::qnumber, math::qnumber, math::qnumber>(),
         py::arg("v"), py::arg("v_rho"), py::arg("v_g1"), py::arg("v_g2"),
         py::arg("v_x"), py::arg("v_y"), py::arg("v_rhorho"), py::arg("v_g1g1"),
         py::arg("v_g2g2"), py::arg("v_xx"), py::arg("v_yy")
    )
    .def_readwrite("v", &ngmix::vDeriv::v)
    .def_readwrite("v_rho", &ngmix::vDeriv::v_rho)
    .def_readwrite("v_g1", &ngmix::vDeriv::v_g1)
    .def_readwrite("v_g2", &ngmix::vDeriv::v_g2)
    .def_readwrite("v_x", &ngmix::vDeriv::v_x)
    .def_readwrite("v_y", &ngmix::vDeriv::v_y)
    .def_readwrite("v_rhorho", &ngmix::vDeriv::v_rhorho)
    .def_readwrite("v_g1g1", &ngmix::vDeriv::v_g1g1)
    .def_readwrite("v_g2g2", &ngmix::vDeriv::v_g2g2)
    .def_readwrite("v_xx", &ngmix::vDeriv::v_xx)
    .def_readwrite("v_yy", &ngmix::vDeriv::v_yy);

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
        .def("apply", &ngmix::NgmixModel::apply,
            "Returns the distorted function value at x, y",
            py::arg("x"), py::arg("y")
        );

    py::class_<ngmix::NgmixGaussian, ngmix::NgmixModel>(ngmix, "NgmixGaussian")
        .def(py::init<double>(),
            "NgmixGaussian basis function",
            py::arg("sigma")
        );

}
}
