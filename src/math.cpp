#include "anacal.h"

namespace anacal {
namespace math {
    void
    pyExportMath(py::module_ &m) {
        py::module_ math = m.def_submodule(
            "math",
            "submodule for Math"
        );
        py::class_<tnumber>(math, "tnumber")
            .def(py::init<>())
            .def(
                py::init<double>(),
                "initialize tnumber with double number as value",
                py::arg("v")
            )
            .def(
                py::init<double, double, double, double, double>(),
                "initialize tnumber with double numbers",
                py::arg("v"), py::arg("g1"), py::arg("g2"),
                py::arg("x1"), py::arg("x2")
            )
            .def(
                py::init<std::array<double, 5>&>(),
                "initialize tnumber with a list of double numbers",
                py::arg("data")
            )
            .def_readwrite("v", &tnumber::v)
            .def_readwrite("g1", &tnumber::g1)
            .def_readwrite("g2", &tnumber::g2)
            .def_readwrite("x1", &tnumber::x1)
            .def_readwrite("x2", &tnumber::x2)
            .def(
                "to_array",
                &tnumber::to_array,
                "Return the numpy array in shape (5)"
            )
            .def("__repr__", [](
                const tnumber &tn) {
                return "<tnumber v=" + std::to_string(tn.v) +
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
        math.def("exp", &exp, "Compute the exponential of a tnumber");
        math.def("pow", &pow, "Compute the exponential of a tnumber");

        py::class_<tmatrix<6, 6>>(math, "tmatrix6")
            .def(py::init<>())
            .def(
                py::init<std::array<std::array<tnumber, 6>, 6>&>(),
                "initialize tmatrix with a 2D list of tnumber",
                py::arg("data")
            )
            .def(
                py::init<std::array<std::array<double, 6>, 6>&>(),
                "initialize tmatrix with a 2D list of double",
                py::arg("data")
            )
            .def(
                py::init<py::array_t<double>&>(),
                "initialize tmatrix with a 2D numpy array",
                py::arg("data")
            )
            .def_readonly("data", &tmatrix<6, 6>::data)
            .def(
                "transpose",
                &tmatrix<6, 6>::transpose,
                "Return the transposed matrix"
            )
            .def(
                "to_array",
                &tmatrix<6, 6>::to_array,
                "Return the numpy array in shape (6, 6, 5)"
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self + tnumber())
            .def(tnumber() + py::self)
            .def(py::self - tnumber())
            .def(tnumber() - py::self)
            .def(py::self * tnumber())
            .def(tnumber() * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(py::self / tnumber())
            .def(py::self * py::self);

        py::class_<tmatrix<8, 8>>(math, "tmatrix8")
            .def(py::init<>())
            .def(
                py::init<std::array<std::array<tnumber, 8>, 8>&>(),
                "initialize tmatrix with a 2D list of tnumber",
                py::arg("data")
            )
            .def(
                py::init<std::array<std::array<double, 8>, 8>&>(),
                "initialize tmatrix with a 2D list of double number",
                py::arg("data")
            )
            .def(
                py::init<py::array_t<double>&>(),
                "initialize tmatrix with a 2D numpy array of double number",
                py::arg("data")
            )
            .def_readonly("data", &tmatrix<8, 8>::data)
            .def(
                "transpose",
                &tmatrix<8, 8>::transpose,
                "Return the transposed matrix"
            )
            .def(
                "to_array",
                &tmatrix<8, 8>::to_array,
                "Return the numpy array in shape (8, 8, 5)"
            )
            .def(-py::self)
            .def(py::self + py::self)
            .def(py::self - py::self)
            .def(py::self + tnumber())
            .def(tnumber() + py::self)
            .def(py::self - tnumber())
            .def(tnumber() - py::self)
            .def(py::self * tnumber())
            .def(tnumber() * py::self)
            .def(py::self * double())
            .def(double() * py::self)
            .def(py::self / double())
            .def(py::self / tnumber())
            .def(py::self * py::self);
        math.def("eye6", &eye<6>, "Generate a indentity tmatrix");
        math.def("eye8", &eye<8>, "Generate a indentity tmatrix");

        py::class_<lossNumber>(math, "lossNumber")
            .def(py::init<
                tnumber, tnumber, tnumber, tnumber,
                tnumber, tnumber, tnumber, tnumber,
                tnumber, tnumber, tnumber, tnumber,
                tnumber>(),
                py::arg("v"),
                py::arg("v_A"), py::arg("v_t"),
                py::arg("v_e1"), py::arg("v_e2"),
                py::arg("v_x1"), py::arg("v_x2"),
                py::arg("v_AA"), py::arg("v_tt"),
                py::arg("v_e1e1"), py::arg("v_e2e2"),
                py::arg("v_x1x1"), py::arg("v_x2x2")
            )
            .def_readwrite("v", &lossNumber::v)
            .def_readwrite("v_A", &lossNumber::v_A)
            .def_readwrite("v_t", &lossNumber::v_t)
            .def_readwrite("v_e1", &lossNumber::v_e1)
            .def_readwrite("v_e2", &lossNumber::v_e2)
            .def_readwrite("v_x1", &lossNumber::v_x1)
            .def_readwrite("v_x2", &lossNumber::v_x2)
            .def_readwrite("v_AA", &lossNumber::v_AA)
            .def_readwrite("v_tt", &lossNumber::v_tt)
            .def_readwrite("v_e1e1", &lossNumber::v_e1e1)
            .def_readwrite("v_e2e2", &lossNumber::v_e2e2)
            .def_readwrite("v_x1x1", &lossNumber::v_x1x1)
            .def_readwrite("v_x2x2", &lossNumber::v_x2x2);

    }
} // math
} // anacal
