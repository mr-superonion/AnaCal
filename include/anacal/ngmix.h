#ifndef ANACAL_NGMIX_H
#define ANACAL_NGMIX_H

#include "ngmix/rmodel.h"
#include "ngmix/fitting.h"

namespace anacal {
namespace ngmix {
    void pyExportNgmix(py::module_& m);
}
}

#endif // ANACAL_NGMIX_H
