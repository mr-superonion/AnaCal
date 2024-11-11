#include "anacal.h"


namespace anacal {

void
pybindFpfsCatalog(py::module_& fpfs) {
    PYBIND11_NUMPY_DTYPE(
        FpfsShapelets,
        m00, m20, m22c, m22s,
        m40, m42c, m42s, m44c, m44s,
        m60, m64c, m64s
    );

    PYBIND11_NUMPY_DTYPE(
        FpfsDetect,
        m00, m20, m22c, m22s,
        m40, m42c, m42s, m44c, m44s,
        m60, m64c, m64s,
        v0, v1, v2, v3,
        dv0_dg1, dv1_dg1, dv2_dg1, dv3_dg1,
        dv0_dg2, dv1_dg2, dv2_dg2, dv3_dg2
    );

    PYBIND11_NUMPY_DTYPE(
        FpfsShapeletsResponse,
        dm00_dg1, dm00_dg2,
        dm20_dg1, dm20_dg2,
        dm22c_dg1, dm22s_dg2,
        dm42c_dg1, dm42s_dg2
    );

    PYBIND11_NUMPY_DTYPE(
        FpfsShape,
        e1,
        de1_dg1,
        e2,
        de2_dg2,
        q1,
        dq1_dg1,
        q2,
        dq2_dg2,
        m00,
        dm00_dg1,
        dm00_dg2,
        m20,
        dm20_dg1,
        dm20_dg2
    );

    PYBIND11_NUMPY_DTYPE(
        FpfsWeight,
        w,
        dw_dg1,
        dw_dg2
    );

    PYBIND11_NUMPY_DTYPE(
        FpfsCatalog,
        e1,
        de1_dg1,
        e2,
        de2_dg2,
        q1,
        dq1_dg1,
        q2,
        dq2_dg2,
        w,
        dw_dg1,
        dw_dg2,
        m00,
        dm00_dg1,
        dm00_dg2,
        m20,
        dm20_dg1,
        dm20_dg2
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

    // Bind measure_shapelets_dg for FpfsDetect
    fpfs.def(
        "measure_shapelets_dg",
        py::overload_cast<
            const FpfsDetect&,
            const std::optional<FpfsDetect>&
        >(&measure_shapelets_dg<FpfsDetect>),
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

    // Bind measure_shapelets_dg for FpfsDetect array
    fpfs.def(
        "measure_shapelets_dg",
        py::overload_cast<
            const py::array_t<FpfsDetect>&,
            const std::optional<py::array_t<FpfsDetect>>&
        >(&measure_shapelets_dg<FpfsDetect>),
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

    // Bind measure_fpfs_shape for FpfsDetect
    fpfs.def(
        "measure_fpfs_shape",
        py::overload_cast<
            double,
            const FpfsDetect&,
            const FpfsShapeletsResponse&
        >(&measure_fpfs_shape<FpfsDetect>),
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

    // Bind measure_fpfs_shape for FpfsDetect array
    fpfs.def(
        "measure_fpfs_shape",
        py::overload_cast<
            double,
            const py::array_t<FpfsDetect>&,
            const py::array_t<FpfsShapeletsResponse>&
        >(&measure_fpfs_shape<FpfsDetect>),
        py::arg("C0"), py::arg("x_array"), py::arg("x_dg_array")
    );

    // Bind measure_fpfs_wsel for FpfsShapelets
    fpfs.def(
        "measure_fpfs_wsel",
        &measure_fpfs_wsel<FpfsShapelets>,
        py::arg("m00_min"), py::arg("std_m00"),
        py::arg("r2_min"), py::arg("omega_r2"),
        py::arg("x"), py::arg("x_dg")
    );

    // Bind measure_fpfs_wsel for FpfsDetect
    fpfs.def(
        "measure_fpfs_wsel",
        &measure_fpfs_wsel<FpfsDetect>,
        py::arg("m00_min"), py::arg("std_m00"),
        py::arg("r2_min"), py::arg("omega_r2"),
        py::arg("x"), py::arg("x_dg")
    );

    // Bind measure_fpfs_wdet0 for FpfsDetect
    fpfs.def(
        "measure_fpfs_wdet0",
        py::overload_cast<
            double,
            double,
            const FpfsDetect&,
            const std::optional<FpfsDetect>&
        >(&measure_fpfs_wdet0),
        py::arg("v_min"),
        py::arg("omega_v"),
        py::arg("x"),
        py::arg("y") = std::nullopt
    );

    // Bind measure_fpfs_wdet0 for FpfsDetect array
    fpfs.def(
        "measure_fpfs_wdet0",
        py::overload_cast<
            double,
            double,
            const py::array_t<FpfsDetect>&,
            const std::optional<py::array_t<FpfsDetect>>&
        >(&measure_fpfs_wdet0),
        py::arg("v_min"),
        py::arg("omega_v"),
        py::arg("x_array"),
        py::arg("y_array") = std::nullopt
    );

    // Bind measure_fpfs_wdet for FpfsDetect
    fpfs.def(
        "measure_fpfs_wdet",
        &measure_fpfs_wdet,
        py::arg("v_min"),
        py::arg("omega_v"),
        py::arg("pthres"),
        py::arg("x"),
        py::arg("y") = std::nullopt
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

    // Overload 2: measure_fpfs for FpfsDetect
    fpfs.def(
        "measure_fpfs",
        py::overload_cast<
            double, double, double, double,
            double, double, double, double,
            const FpfsDetect&,
            const std::optional<FpfsDetect>&
        >(&measure_fpfs),
        py::arg("C0"),
        py::arg("v_min"),
        py::arg("omega_v"),
        py::arg("pthres"),
        py::arg("m00_min"),
        py::arg("std_m00"),
        py::arg("r2_min"),
        py::arg("omega_r2"),
        py::arg("x"),
        py::arg("y") = std::nullopt
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

    // Overload 4: measure_fpfs for FpfsDetect (array version)
    fpfs.def(
        "measure_fpfs",
        py::overload_cast<
            double, double, double, double, double, double, double, double,
            const py::array_t<FpfsDetect>&,
            const std::optional<py::array_t<FpfsDetect>>&
        >(&measure_fpfs),
        py::arg("C0"),
        py::arg("v_min"),
        py::arg("omega_v"),
        py::arg("pthres"),
        py::arg("m00_min"),
        py::arg("std_m00"),
        py::arg("r2_min"),
        py::arg("omega_r2"),
        py::arg("x_array"),
        py::arg("y_array") = std::nullopt
    );
}
}
