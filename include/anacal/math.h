#ifndef ANACAL_MATH_H
#define ANACAL_MATH_H

#include "math/tnumber.h"
#include "math/tmatrix.h"
#include "math/pooling.h"
#include "math/loss.h"
#include <pybind11/operators.h>

namespace anacal {
namespace math {
    void pyExportMath(py::module_& m);
}
}

#endif // ANACAL_MATH_H
