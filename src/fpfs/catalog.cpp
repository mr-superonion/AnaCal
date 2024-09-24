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
        v0_g1, v1_g1, v2_g1, v3_g1,
        v0_g2, v1_g2, v2_g2, v3_g2
    );

    PYBIND11_NUMPY_DTYPE(
        FpfsShapeletsResponse,
        m00_g1, m00_g2,
        m20_g1, m20_g2, m22c_g1, m22s_g2,
        m42c_g1, m42s_g2
    );
    /* fpfs.def( */
    /*     "calculate_shapelets_dg", */
    /*     py::overload_cast<const FpfsShapelets&>(&calculate_shapelets_dg), */
    /*     "Compute shear response for FpfsShapelets" */
    /* ); */
    /* fpfs.def( */
    /*     "calculate_shapelets_dg", */
    /*     py::overload_cast<const FpfsDetect&>(&calculate_shapelets_dg), */
    /*     "Compute shear response for FpfsDetect" */
    /* ); */
}
}
