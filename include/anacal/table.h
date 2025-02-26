#ifndef ANACAL_TABLE_H
#define ANACAL_TABLE_H

#include "stdafx.h"
#include "math.h"

namespace anacal {
namespace table {

struct galRow{
    double flux;
    double dflux_dg1;
    double dflux_dg2;
    double rho;
    double drho_dg1;
    double drho_dg2;
    double e1;
    double de1_dg1;
    double de1_dg2;
    double e2;
    double de2_dg1;
    double de2_dg2;
    double x;
    double dx_dg1;
    double dx_dg2;
    double y;
    double dy_dg1;
    double dy_dg2;
    double wdet;
    double dwdet_dg1;
    double dwdet_dg2;
    int mask_value;
    bool is_peak;
};

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

    inline galRow
    to_row() const {
        math::tnumber rho = math::exp(model.A);
        galRow row = {
            model.A.v,
            model.A.g1,
            model.A.g2,
            rho.v,
            rho.g1,
            rho.g2,
            model.e1.v,
            model.e1.g1,
            model.e1.g2,
            model.e2.v,
            model.e2.g1,
            model.e2.g2,
            model.x1.v,
            model.x1.g1,
            model.x1.g2,
            model.x2.v,
            model.x2.g1,
            model.x2.g2,
            wdet.v,
            wdet.g1,
            wdet.g2,
            mask_value,
            is_peak
        };
        return row;
    };
};

inline py::array_t<galRow>
objlist_to_array(
    const std::vector<galNumber> & catalog
) {
    int nrow = catalog.size();
    py::array_t<galRow> result(nrow);
    auto r_r = result.mutable_unchecked<1>();
    for (ssize_t j = 0; j < nrow; ++j) {
        r_r(j) = catalog[j].to_row();
    }
    return result;
};

void pyExportTable(py::module_& m);

} // table
} // anacal
#endif // ANACAL_TABLE_H
