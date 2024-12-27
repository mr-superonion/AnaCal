#include "anacal.h"

namespace anacal {
    void
    pyExportMath(py::module_ &m) {
        py::module_ math = m.def_submodule(
            "math",
            "submodule for Math"
        );
        py::class_<math::qnumber>(math, "qnumber")
            .def(
                py::init<double, double, double, double, double>(),
                py::arg("v"), py::arg("g1"), py::arg("g2"),
                py::arg("x1"), py::arg("x2")
            )
            .def_readwrite("v", &math::qnumber::v)
            .def_readwrite("g1", &math::qnumber::g1)
            .def_readwrite("g2", &math::qnumber::g2)
            .def_readwrite("x1", &math::qnumber::x1)
            .def_readwrite("x2", &math::qnumber::x2)
            .def("__repr__", [](
                const math::qnumber &q) {
                return "<qnumber v=" + std::to_string(q.v) +
                       ", g1=" + std::to_string(q.g1) +
                       ", g2=" + std::to_string(q.g2) +
                       ", x1=" + std::to_string(q.x1) +
                       ", x2=" + std::to_string(q.x2) + ">";
                }
            )
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self * py::self)
            .def(py::self / py::self)
            .def(py::self + double())
            .def(double() + py::self)
            .def(py::self - double())
            .def(double() - py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(double() / py::self);
        math.def("exp", &math::exp, "Compute the exponential of a qnumber");
    }
}
