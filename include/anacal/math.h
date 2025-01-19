#ifndef ANACAL_MATH_H
#define ANACAL_MATH_H

#include "math/qnumber.h"
#include "math/qmatrix.h"
#include "math/pooling.h"
#include <pybind11/operators.h>

namespace anacal {
    void pyExportMath(py::module_& m);
}

#endif // ANACAL_MATH_H
