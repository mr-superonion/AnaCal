#ifndef ANACAL_TABLE_H
#define ANACAL_TABLE_H

#include "stdafx.h"
#include "math.h"
#include "geometry.h"

namespace anacal {
namespace table {

struct galRow{
    double ra;
    double dec;
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
    double fluxap2;
    double dfluxap2_dg1;
    double dfluxap2_dg2;
    double wdet;
    double dwdet_dg1;
    double dwdet_dg2;
    int mask_value;
    bool is_peak;
    bool is_primary;
    double fpfs_e1;
    double fpfs_de1_dg1;
    double fpfs_de1_dg2;
    double fpfs_e2;
    double fpfs_de2_dg1;
    double fpfs_de2_dg2;
    double fpfs_m0;
    double fpfs_dm0_dg1;
    double fpfs_dm0_dg2;
    double fpfs_m2;
    double fpfs_dm2_dg1;
    double fpfs_dm2_dg2;
    double peakv;
    double dpeakv_dg1;
    double dpeakv_dg2;
    double bkg;
    double dbkg_dg1;
    double dbkg_dg2;
};

struct galNumber {
    // value with derivatives to Gaussian model parameters
    ngmix::NgmixGaussian model;
    math::qnumber fluxap2;
    math::qnumber wdet;
    int mask_value=0;
    bool is_peak=false;
    bool is_primary=true;
    math::lossNumber loss;
    math::qnumber fpfs_e1;
    math::qnumber fpfs_e2;
    math::qnumber fpfs_m0;
    math::qnumber fpfs_m2;
    math::qnumber peakv;
    math::qnumber bkg;
    double ra = 0.0;
    double dec = 0.0;

    galNumber() = default;

    galNumber(
        ngmix::NgmixGaussian model,
        math::qnumber fluxap2,
        math::qnumber wdet,
        int mask_value,
        bool is_peak,
        math::lossNumber loss
    ) : model(model), fluxap2(fluxap2), wdet(wdet),
        mask_value(mask_value), is_peak(is_peak), loss(loss) {}

    inline galNumber
    decentralize(const geometry::block & block) const {
        double dx1 = this->model.x1.v - block.xcen * block.scale;
        double dx2 = this->model.x2.v - block.ycen * block.scale;
        // (dx1, dx2) is the position of the source wrt center of block
        galNumber result= *this;
        result.wdet = this->wdet.decentralize(dx1, dx2);
        result.fluxap2 = this->fluxap2.decentralize(dx1, dx2);
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
        result.fluxap2 = this->fluxap2.centralize(dx1, dx2);
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
            ra,
            dec,
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
            fluxap2.v,
            fluxap2.g1,
            fluxap2.g2,
            wdet.v,
            wdet.g1,
            wdet.g2,
            mask_value,
            is_peak,
            is_primary,
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
            fpfs_m2.g2,
            peakv.v,
            peakv.g1,
            peakv.g2,
            bkg.v,
            bkg.g1,
            bkg.g2
        };
        return row;
    };

    inline void
    from_row(galRow & row) {
        ra = row.ra;
        dec = row.dec;
        model.F = math::qnumber(row.flux, row.dflux_dg1, row.dflux_dg2);
        model.t = math::qnumber(row.t, row.dt_dg1, row.dt_dg2);
        model.a1 = math::qnumber(row.a1, row.da1_dg1, row.da1_dg2);
        model.a2 = math::qnumber(row.a2, row.da2_dg1, row.da2_dg2);
        model.x1 = math::qnumber(row.x1, row.dx1_dg1, row.dx1_dg2);
        model.x2= math::qnumber(row.x2, row.dx2_dg1, row.dx2_dg2);
        fluxap2 = math::qnumber(
            row.fluxap2, row.dfluxap2_dg1, row.dfluxap2_dg2
        );
        wdet = math::qnumber(row.wdet, row.dwdet_dg1, row.dwdet_dg2);
        mask_value = row.mask_value;
        is_peak = row.is_peak;
        is_primary = row.is_primary;
        fpfs_e1 = math::qnumber(
            row.fpfs_e1, row.fpfs_de1_dg1, row.fpfs_de1_dg2
        );
        fpfs_e2 = math::qnumber(
            row.fpfs_e2, row.fpfs_de2_dg1, row.fpfs_de2_dg2
        );
        fpfs_m0 = math::qnumber(
            row.fpfs_m0, row.fpfs_dm0_dg1, row.fpfs_dm0_dg2
        );
        fpfs_m2 = math::qnumber(
            row.fpfs_m2, row.fpfs_dm2_dg1, row.fpfs_dm2_dg2
        );
        peakv = math::qnumber(row.peakv, row.dpeakv_dg1, row.dpeakv_dg2);
        bkg = math::qnumber(row.bkg, row.dbkg_dg1, row.dbkg_dg2);
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
