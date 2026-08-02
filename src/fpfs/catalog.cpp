#include "anacal.h"


namespace anacal {
    namespace fpfs {

void
pyExportFpfsCatalog(py::module_& fpfs) {
    PYBIND11_NUMPY_DTYPE(
        FpfsShapelets,
        m00, m20, m22c, m22s,
        m40, m42c, m42s, m44c, m44s,
        m60, m64c, m64s
    );

    PYBIND11_NUMPY_DTYPE(
        FpfsShapeletsResponse,
        dm00_dg1, dm00_dg2,
        dm20_dg1, dm20_dg2,
        dm22c_dg1, dm22c_dg2,
        dm22s_dg1, dm22s_dg2,
        dm42c_dg1, dm42c_dg2,
        dm42s_dg1, dm42s_dg2
    );

    PYBIND11_NUMPY_DTYPE(
        FpfsShape,
        e1,
        de1_dg1,
        de1_dg2,
        e2,
        de2_dg1,
        de2_dg2,
        q1,
        dq1_dg1,
        dq1_dg2,
        q2,
        dq2_dg1,
        dq2_dg2,
        m00,
        dm00_dg1,
        dm00_dg2,
        m20,
        dm20_dg1,
        dm20_dg2,
        m22c,
        dm22c_dg1,
        dm22c_dg2,
        m22s,
        dm22s_dg1,
        dm22s_dg2,
        m42c,
        dm42c_dg1,
        dm42c_dg2,
        m42s,
        dm42s_dg1,
        dm42s_dg2
    );

    // Bind measure_shapelets_dg for FpfsShapelets
    fpfs.def(
        "measure_shapelets_dg",
        py::overload_cast<
            const FpfsShapelets&,
            const std::optional<FpfsShapelets>&
        >(&measure_shapelets_dg<FpfsShapelets>),
        py::arg("x"), py::arg("y") = std::nullopt
    );

    // Bind measure_shapelets_dg for FpfsShapelets array
    fpfs.def(
        "measure_shapelets_dg",
        py::overload_cast<
            const py::array_t<FpfsShapelets>&,
            const std::optional<py::array_t<FpfsShapelets>>&
        >(&measure_shapelets_dg<FpfsShapelets>),
        py::arg("x_array"), py::arg("y_array") = std::nullopt
    );

    // Bind measure_fpfs_shape for FpfsShapelets
    fpfs.def(
        "measure_fpfs_shape",
        py::overload_cast<
            double,
            const FpfsShapelets&,
            const FpfsShapeletsResponse&
        >(&measure_fpfs_shape<FpfsShapelets>),
        py::arg("C0"), py::arg("x"), py::arg("x_dg")
    );

    // Bind measure_fpfs_shape for FpfsShapelets array
    fpfs.def(
        "measure_fpfs_shape",
        py::overload_cast<
            double,
            const py::array_t<FpfsShapelets>&,
            const py::array_t<FpfsShapeletsResponse>&
        >(&measure_fpfs_shape<FpfsShapelets>),
        py::arg("C0"), py::arg("x_array"), py::arg("x_dg_array")
    );

     // Overload 1: measure_fpfs for FpfsShapelets
    fpfs.def(
        "measure_fpfs",
        py::overload_cast<
            double, const FpfsShapelets&,
            const std::optional<FpfsShapelets>&
        >(&measure_fpfs),
        py::arg("C0"), py::arg("x"), py::arg("y") = std::nullopt
    );

    // Overload 3: measure_fpfs for FpfsShapelets (array version)
    fpfs.def(
        "measure_fpfs",
        py::overload_cast<
            double, const py::array_t<FpfsShapelets>&,
            const std::optional<py::array_t<FpfsShapelets>>&
        >(&measure_fpfs),
        py::arg("C0"),
        py::arg("x_array"),
        py::arg("y_array") = std::nullopt
    );

}
    } // namespace fpfs
} // namespace anacal
