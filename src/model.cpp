#include "anacal.h"


namespace anacal {

void pyExportModel(py::module& m) {
    py::module_ model = m.def_submodule("model", "submodule for models");
    py::class_<BaseModel>(model, "BaseModel")
        .def(py::init<>())
        .def("apply", &BaseModel::apply,
            "Returns the pre-distorted function value at kx, ky",
            py::arg("kx"), py::arg("ky")
        )
        .def("fValue", &BaseModel::fValue,
            "Returns the distorted value at kx, ky",
            py::arg("kx"), py::arg("ky")
        )
        .def("set_transform", &BaseModel::set_transform,
            "Sets the transform parameters",
            py::arg("theta"), py::arg("gamma1"), py::arg("gamma2")
        )
        .def("draw", &BaseModel::draw,
            "draw the distorted function to ndarray",
            py::arg("scale"), py::arg("nx"), py::arg("ny")
        );

    py::class_<Gaussian, BaseModel>(model, "Gaussian")
        .def(py::init<double>(),
            "Gaussian basis function",
            py::arg("sigma")
        );

    py::class_<GaussianG1, BaseModel>(model, "GaussianG1")
        .def(py::init<double>(),
            "Gaussian's g1 multiplier Function",
            py::arg("sigma")
        );

    py::class_<GaussianG2, BaseModel>(model, "GaussianG2")
        .def(py::init<double>(),
            "Gaussian's g2 multiplier Function",
            py::arg("sigma")
        );

    py::class_<GaussianX1, BaseModel>(model, "GaussianX1")
        .def(py::init<>(),
            "Gaussian's x1 multiplier Function"
        );

    py::class_<GaussianX2, BaseModel>(model, "GaussianX2")
        .def(py::init<>(),
            "Gaussian's x2 multiplier Function"
        );

    py::class_<GaussianTopHat, BaseModel>(model, "GaussianTopHat")
        .def(py::init<double, double>(),
            "Gaussian convolved with top-hat basis function",
            py::arg("d"), py::arg("sigma")
        );

    py::class_<GaussianTopHat2, BaseModel>(model, "GaussianTopHat2")
        .def(py::init<double, double>(),
            "Square ofGaussian convolved with top-hat basis function",
            py::arg("d"), py::arg("sigma")
        );

    py::class_<CirTopHat, BaseModel>(model, "CirTopHat")
        .def(py::init<double>(),
            "Circular top-hat basis function",
            py::arg("d")
        );
}

}
