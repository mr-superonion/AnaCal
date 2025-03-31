#ifndef ANACAL_TABLE_H
#define ANACAL_TABLE_H

#include "stdafx.h"
#include "math.h"
#include "geometry.h"

namespace anacal {
namespace table {

struct galRow{
    double flux;
    double dflux_dg1;
    double dflux_dg2;
    double t;
    double dt_dg1;
    double dt_dg2;
    double a1;
    double da1_dg1;
    double da1_dg2;
    double a2;
    double da2_dg1;
    double da2_dg2;
    double e1;
    double de1_dg1;
    double de1_dg2;
    double e2;
    double de2_dg1;
    double de2_dg2;
    double x1;
    double dx1_dg1;
    double dx1_dg2;
    double x2;
    double dx2_dg1;
    double dx2_dg2;
    double fluxdet;
    double dfluxdet_dg1;
    double dfluxdet_dg2;
    double wdet;
    double dwdet_dg1;
    double dwdet_dg2;
    int mask_value;
    bool is_peak;
    double fpfs_e1;
    double fpfs_de1_dg1;
    double fpfs_de1_dg2;
    double fpfs_e2;
    double fpfs_de2_dg1;
    double fpfs_de2_dg2;
    double fpfs_m0;
    double fpfs_m0_dg1;
    double fpfs_m0_dg2;
    double fpfs_m2;
    double fpfs_m2_dg1;
    double fpfs_m2_dg2;
};

struct galNumber {
    // value with derivatives to Gaussian model parameters
    ngmix::NgmixGaussian model;
    math::qnumber fluxdet;
    math::qnumber wdet;
    int mask_value=0;
    bool is_peak=false;
    math::lossNumber loss;
    math::qnumber fpfs_e1;
    math::qnumber fpfs_e2;
    math::qnumber fpfs_m0;
    math::qnumber fpfs_m2;

    galNumber() = default;

    galNumber(
        ngmix::NgmixGaussian model,
        math::qnumber fluxdet,
        math::qnumber wdet,
        int mask_value,
        bool is_peak,
        math::lossNumber loss
    ) : model(model), fluxdet(fluxdet), wdet(wdet),
        mask_value(mask_value), is_peak(is_peak), loss(loss) {}

    inline galNumber
    decentralize(const geometry::block & block) const {
        double dx1 = this->model.x1.v - block.xcen * block.scale;
        double dx2 = this->model.x2.v - block.ycen * block.scale;
        // (dx1, dx2) is the position of the source wrt center of block
        galNumber result= *this;
        result.wdet = this->wdet.decentralize(dx1, dx2);
        result.fluxdet = this->fluxdet.decentralize(dx1, dx2);
        result.model = this->model.decentralize(dx1, dx2);
        result.fpfs_e1 = this->fpfs_e1.decentralize(dx1, dx2);
        result.fpfs_e2 = this->fpfs_e2.decentralize(dx1, dx2);
        result.fpfs_m0 = this->fpfs_m0.decentralize(dx1, dx2);
        result.fpfs_m2 = this->fpfs_m2.decentralize(dx1, dx2);
        return result;
    };

    inline galNumber
    centralize(const geometry::block & block) const {
        double dx1 = this->model.x1.v - block.xcen * block.scale;
        double dx2 = this->model.x2.v - block.ycen * block.scale;
        // (dx1, dx2) is the position of the source wrt center of block
        galNumber result= *this;
        result.wdet = this->wdet.centralize(dx1, dx2);
        result.fluxdet = this->fluxdet.centralize(dx1, dx2);
        result.model = this->model.centralize(dx1, dx2);
        result.fpfs_e1 = this->fpfs_e1.centralize(dx1, dx2);
        result.fpfs_e2 = this->fpfs_e2.centralize(dx1, dx2);
        result.fpfs_m0 = this->fpfs_m0.centralize(dx1, dx2);
        result.fpfs_m2 = this->fpfs_m2.centralize(dx1, dx2);
        return result;
    };

    inline galRow
    to_row() const {
        std::array<math::qnumber, 2> shape = model.get_shape();
        galRow row = {
            model.F.v,
            model.F.g1,
            model.F.g2,
            model.t.v,
            model.t.g1,
            model.t.g2,
            model.a1.v,
            model.a1.g1,
            model.a1.g2,
            model.a2.v,
            model.a2.g1,
            model.a2.g2,
            shape[0].v,
            shape[0].g1,
            shape[0].g2,
            shape[1].v,
            shape[1].g1,
            shape[1].g2,
            model.x1.v,
            model.x1.g1,
            model.x1.g2,
            model.x2.v,
            model.x2.g1,
            model.x2.g2,
            fluxdet.v,
            fluxdet.g1,
            fluxdet.g2,
            wdet.v,
            wdet.g1,
            wdet.g2,
            mask_value,
            is_peak,
            fpfs_e1.v,
            fpfs_e1.g1,
            fpfs_e1.g2,
            fpfs_e2.v,
            fpfs_e2.g1,
            fpfs_e2.g2,
            fpfs_m0.v,
            fpfs_m0.g1,
            fpfs_m0.g2,
            fpfs_m2.v,
            fpfs_m2.g1,
            fpfs_m2.g2
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
