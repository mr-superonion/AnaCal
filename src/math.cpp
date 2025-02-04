#include "anacal.h"

namespace anacal {
    void
    pyExportMath(py::module_ &m) {
        py::module_ math = m.def_submodule(
            "math",
            "submodule for Math"
        );
        py::class_<math::qnumber>(math, "qnumber")
            .def(py::init<>())
            .def(
                py::init<double, double, double, double, double>(),
                py::arg("v"), py::arg("g1"), py::arg("g2"),
                py::arg("x1"), py::arg("x2")
            )
            .def(
                py::init<std::array<double, 5>&>(),
                "initialize qnumber with a list of double numbers",
                py::arg("data")
            )
            .def_readonly("v", &math::qnumber::v)
            .def_readonly("g1", &math::qnumber::g1)
            .def_readonly("g2", &math::qnumber::g2)
            .def_readonly("x1", &math::qnumber::x1)
            .def_readonly("x2", &math::qnumber::x2)
            .def(
                "to_array",
                &math::qnumber::to_array,
                "Return the numpy array in shape (5)"
            )
            .def("__repr__", [](
                const math::qnumber &q) {
                return "<qnumber v=" + std::to_string(q.v) +
                       ", g1=" + std::to_string(q.g1) +
                       ", g2=" + std::to_string(q.g2) +
                       ", x1=" + std::to_string(q.x1) +
                       ", x2=" + std::to_string(q.x2) + ">";
                }
            )
            .def(-py::self)
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
        math.def("pow", &math::pow, "Compute the exponential of a qnumber");

        py::class_<math::qmatrix<6, 6>>(math, "qmatrix6")
            .def(py::init<>())
            .def(
                py::init<std::array<std::array<math::qnumber, 6>, 6>&>(),
                "initialize qmatrix with a 2D list of qnumber",
                py::arg("data")
            )
            .def(
                py::init<std::array<std::array<double, 6>, 6>&>(),
                "initialize qmatrix with a 2D list of double",
                py::arg("data")
            )
            .def(
                py::init<py::array_t<double>&>(),
                "initialize qmatrix with a 2D numpy array",
                py::arg("data")
            )
            .def_readonly("data", &math::qmatrix<6, 6>::data)
            .def(
                "transpose",
                &math::qmatrix<6, 6>::transpose,
                "Return the transposed matrix"
            )
            .def(
                "to_array",
                &math::qmatrix<6, 6>::to_array,
                "Return the numpy array in shape (6, 6, 5)"
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self + math::qnumber())
            .def(math::qnumber() + py::self)
            .def(py::self - math::qnumber())
            .def(math::qnumber() - py::self)
            .def(py::self * math::qnumber())
            .def(math::qnumber() * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(py::self / math::qnumber())
            .def(py::self * py::self);

        py::class_<math::qmatrix<8, 8>>(math, "qmatrix8")
            .def(py::init<>())
            .def(
                py::init<std::array<std::array<math::qnumber, 8>, 8>&>(),
                "initialize qmatrix with a 2D list of qnumber",
                py::arg("data")
            )
            .def(
                py::init<std::array<std::array<double, 8>, 8>&>(),
                "initialize qmatrix with a 2D list of double number",
                py::arg("data")
            )
            .def(
                py::init<py::array_t<double>&>(),
                "initialize qmatrix with a 2D numpy array of double number",
                py::arg("data")
            )
            .def_readonly("data", &math::qmatrix<8, 8>::data)
            .def(
                "transpose",
                &math::qmatrix<8, 8>::transpose,
                "Return the transposed matrix"
            )
            .def(
                "to_array",
                &math::qmatrix<8, 8>::to_array,
                "Return the numpy array in shape (8, 8, 5)"
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self + math::qnumber())
            .def(math::qnumber() + py::self)
            .def(py::self - math::qnumber())
            .def(math::qnumber() - py::self)
            .def(py::self * math::qnumber())
            .def(math::qnumber() * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(py::self / math::qnumber())
            .def(py::self * py::self);
        math.def("eye6", &math::eye<6>, "Generate a indentity qmatrix");
        math.def("eye8", &math::eye<8>, "Generate a indentity qmatrix");
    }
}
