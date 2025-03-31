#include "anacal.h"

namespace anacal {
namespace math {
    void
    pyExportMath(py::module_ &m) {
        py::module_ math = m.def_submodule(
            "math",
            "submodule for Math"
        );
        py::class_<qnumber>(math, "qnumber")
            .def(py::init<>())
            .def(
                py::init<double>(),
                "initialize qnumber with double number as value",
                py::arg("v")
            )
            .def(
                py::init<double, double, double, double, double>(),
                "initialize qnumber with double numbers",
                py::arg("v"), py::arg("g1"), py::arg("g2"),
                py::arg("x1"), py::arg("x2")
            )
            .def(
                py::init<std::array<double, 5>&>(),
                "initialize qnumber with a list of double numbers",
                py::arg("data")
            )
            .def_readwrite("v", &qnumber::v)
            .def_readwrite("g1", &qnumber::g1)
            .def_readwrite("g2", &qnumber::g2)
            .def_readwrite("x1", &qnumber::x1)
            .def_readwrite("x2", &qnumber::x2)
            .def(
                "to_array",
                &qnumber::to_array,
                "Return the numpy array in shape (5)"
            )
            .def("__repr__", [](
                const qnumber &tn) {
                return "<qnumber v=" + std::to_string(tn.v) +
                       ", g1=" + std::to_string(tn.g1) +
                       ", g2=" + std::to_string(tn.g2) +
                       ", x1=" + std::to_string(tn.x1) +
                       ", x2=" + std::to_string(tn.x2) + ">";
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
        math.def("exp", &exp, "Compute the exponential of a qnumber");
        math.def("pow", &pow, "Compute the exponential of a qnumber");

        py::class_<qmatrix<6, 6>>(math, "qmatrix6")
            .def(py::init<>())
            .def(
                py::init<std::array<std::array<qnumber, 6>, 6>&>(),
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
            .def_readonly("data", &qmatrix<6, 6>::data)
            .def(
                "transpose",
                &qmatrix<6, 6>::transpose,
                "Return the transposed matrix"
            )
            .def(
                "to_array",
                &qmatrix<6, 6>::to_array,
                "Return the numpy array in shape (6, 6, 5)"
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self + qnumber())
            .def(qnumber() + py::self)
            .def(py::self - qnumber())
            .def(qnumber() - py::self)
            .def(py::self * qnumber())
            .def(qnumber() * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(py::self / qnumber())
            .def(py::self * py::self);

        py::class_<qmatrix<8, 8>>(math, "qmatrix8")
            .def(py::init<>())
            .def(
                py::init<std::array<std::array<qnumber, 8>, 8>&>(),
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
            .def_readonly("data", &qmatrix<8, 8>::data)
            .def(
                "transpose",
                &qmatrix<8, 8>::transpose,
                "Return the transposed matrix"
            )
            .def(
                "to_array",
                &qmatrix<8, 8>::to_array,
                "Return the numpy array in shape (8, 8, 5)"
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self + qnumber())
            .def(qnumber() + py::self)
            .def(py::self - qnumber())
            .def(qnumber() - py::self)
            .def(py::self * qnumber())
            .def(qnumber() * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(py::self / qnumber())
            .def(py::self * py::self);
        math.def("eye6", &eye<6>, "Generate a indentity qmatrix");
        math.def("eye8", &eye<8>, "Generate a indentity qmatrix");

        py::class_<lossNumber>(math, "lossNumber")
            .def(py::init<
                qnumber, qnumber, qnumber, qnumber,
                qnumber, qnumber, qnumber, qnumber,
                qnumber, qnumber, qnumber, qnumber,
                qnumber>(),
                py::arg("v"),
                py::arg("v_F"), py::arg("v_t"),
                py::arg("v_a1"), py::arg("v_a2"),
                py::arg("v_x1"), py::arg("v_x2"),
                py::arg("v_FF"), py::arg("v_tt"),
                py::arg("v_a1a1"), py::arg("v_a2a2"),
                py::arg("v_x1x1"), py::arg("v_x2x2")
            )
            .def_readwrite("v", &lossNumber::v)
            .def_readwrite("v_F", &lossNumber::v_F)
            .def_readwrite("v_t", &lossNumber::v_t)
            .def_readwrite("v_a1", &lossNumber::v_a1)
            .def_readwrite("v_a2", &lossNumber::v_a2)
            .def_readwrite("v_x1", &lossNumber::v_x1)
            .def_readwrite("v_x2", &lossNumber::v_x2)
            .def_readwrite("v_FF", &lossNumber::v_FF)
            .def_readwrite("v_tt", &lossNumber::v_tt)
            .def_readwrite("v_a1a1", &lossNumber::v_a1a1)
            .def_readwrite("v_a2a2", &lossNumber::v_a2a2)
            .def_readwrite("v_x1x1", &lossNumber::v_x1x1)
            .def_readwrite("v_x2x2", &lossNumber::v_x2x2);

    }
} // math
} // anacal
