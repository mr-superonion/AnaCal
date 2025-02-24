#ifndef ANACAL_TABLE_H
#define ANACAL_TABLE_H

#include "stdafx.h"
#include "math.h"

namespace anacal {
namespace table {

struct galNumber {
    // value with derivatives to Gaussian model parameters
    ngmix::NgmixGaussian model;
    math::tnumber wdet;
    int mask_value=0;
    bool is_peak=false;
    math::lossNumber loss;

    galNumber() = default;

    galNumber(
        ngmix::NgmixGaussian model,
        math::tnumber wdet,
        int mask_value,
        bool is_peak,
        math::lossNumber loss
    ) : model(model), wdet(wdet), mask_value(mask_value),
        is_peak(is_peak), loss(loss) {}
};

void pyExportTable(py::module_& m);

} // table
} // anacal
#endif // ANACAL_TABLE_H
