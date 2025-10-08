#ifndef ANACAL_TABLE_H
#define ANACAL_TABLE_H

#include <stdexcept>

#include "math.h"
#include "geometry.h"
#include "ngmix/rmodel.h"

namespace anacal {
namespace table {

struct galRow{
    double ra;
    double dec;
    double flux;
    double dflux_dg1;
    double dflux_dg2;
    double dflux_dj1;
    double dflux_dj2;
    double t;
    double dt_dg1;
    double dt_dg2;
    double dt_dj1;
    double dt_dj2;
    double a1;
    double da1_dg1;
    double da1_dg2;
    double da1_dj1;
    double da1_dj2;
    double a2;
    double da2_dg1;
    double da2_dg2;
    double da2_dj1;
    double da2_dj2;
    double e1;
    double de1_dg1;
    double de1_dg2;
    double de1_dj1;
    double de1_dj2;
    double e2;
    double de2_dg1;
    double de2_dg2;
    double de2_dj1;
    double de2_dj2;
    double x1;
    double dx1_dg1;
    double dx1_dg2;
    double dx1_dj1;
    double dx1_dj2;
    double x2;
    double dx2_dg1;
    double dx2_dg2;
    double dx2_dj1;
    double dx2_dj2;
    double wdet;
    double dwdet_dg1;
    double dwdet_dg2;
    double dwdet_dj1;
    double dwdet_dj2;
    double wsel;
    double dwsel_dg1;
    double dwsel_dg2;
    double dwsel_dj1;
    double dwsel_dj2;
    int mask_value;
    bool is_peak;
    bool is_primary;
    double flux_gauss0;
    double dflux_gauss0_dg1;
    double dflux_gauss0_dg2;
    double dflux_gauss0_dj1;
    double dflux_gauss0_dj2;
    double flux_gauss2;
    double dflux_gauss2_dg1;
    double dflux_gauss2_dg2;
    double dflux_gauss2_dj1;
    double dflux_gauss2_dj2;
    double flux_gauss4;
    double dflux_gauss4_dg1;
    double dflux_gauss4_dg2;
    double dflux_gauss4_dj1;
    double dflux_gauss4_dj2;
    double flux_gauss0_err;
    double flux_gauss2_err;
    double flux_gauss4_err;
    double fpfs_e1;
    double fpfs_de1_dg1;
    double fpfs_de1_dg2;
    double fpfs_de1_dj1;
    double fpfs_de1_dj2;
    double fpfs_e2;
    double fpfs_de2_dg1;
    double fpfs_de2_dg2;
    double fpfs_de2_dj1;
    double fpfs_de2_dj2;
    double fpfs_m0;
    double fpfs_dm0_dg1;
    double fpfs_dm0_dg2;
    double fpfs_dm0_dj1;
    double fpfs_dm0_dj2;
    double fpfs_m2;
    double fpfs_dm2_dg1;
    double fpfs_dm2_dg2;
    double fpfs_dm2_dj1;
    double fpfs_dm2_dj2;
    double x1_det;
    double x2_det;
    int block_id;
};

struct galNumber {
    // value with derivatives to Gaussian model parameters
    ngmix::NgmixGaussian model;
    math::qnumber wdet = math::qnumber(1.0);
    math::qnumber wsel = math::qnumber(1.0);
    int mask_value=0;
    bool is_peak=false;
    bool is_primary=true;
    bool initialized=false;
    math::lossNumber loss;
    math::qnumber fpfs_e1;
    math::qnumber fpfs_e2;
    math::qnumber fpfs_m0;
    math::qnumber fpfs_m2;
    math::qnumber flux_gauss0 = math::qnumber(0.0);
    math::qnumber flux_gauss2 = math::qnumber(0.0);
    math::qnumber flux_gauss4 = math::qnumber(0.0);
    double flux_gauss0_err = 0.0;
    double flux_gauss2_err = 0.0;
    double flux_gauss4_err = 0.0;
    double ra = 0.0;
    double dec = 0.0;
    double x1_det, x2_det;
    int block_id;

    galNumber() = default;

    galNumber(
        ngmix::NgmixGaussian model,
        math::qnumber wdet,
        int mask_value,
        bool is_peak,
        math::lossNumber loss
    ) : model(model), wdet(wdet),
        mask_value(mask_value), is_peak(is_peak), loss(loss) {}

    inline galNumber
    decentralize(const geometry::block & block) const {
        double dx1 = this->x1_det - block.xcen * block.scale;
        double dx2 = this->x2_det - block.ycen * block.scale;
        // (dx1, dx2) is the position of the source wrt center of block
        galNumber result= *this;
        result.wdet = this->wdet.decentralize(dx1, dx2);
        result.wsel = this->wsel.decentralize(dx1, dx2);
        result.model = this->model.decentralize(dx1, dx2);
        result.fpfs_e1 = this->fpfs_e1.decentralize(dx1, dx2);
        result.fpfs_e2 = this->fpfs_e2.decentralize(dx1, dx2);
        result.fpfs_m0 = this->fpfs_m0.decentralize(dx1, dx2);
        result.fpfs_m2 = this->fpfs_m2.decentralize(dx1, dx2);
        result.flux_gauss0 = this->flux_gauss0.decentralize(dx1, dx2);
        result.flux_gauss2 = this->flux_gauss2.decentralize(dx1, dx2);
        result.flux_gauss4 = this->flux_gauss4.decentralize(dx1, dx2);
        return result;
    };

    inline galNumber
    centralize(const geometry::block & block) const {
        double dx1 = this->x1_det - block.xcen * block.scale;
        double dx2 = this->x2_det - block.ycen * block.scale;
        // (dx1, dx2) is the position of the source wrt center of block
        galNumber result= *this;
        result.wdet = this->wdet.centralize(dx1, dx2);
        result.wsel = this->wsel.centralize(dx1, dx2);
        result.model = this->model.centralize(dx1, dx2);
        result.fpfs_e1 = this->fpfs_e1.centralize(dx1, dx2);
        result.fpfs_e2 = this->fpfs_e2.centralize(dx1, dx2);
        result.fpfs_m0 = this->fpfs_m0.centralize(dx1, dx2);
        result.fpfs_m2 = this->fpfs_m2.centralize(dx1, dx2);
        result.flux_gauss0 = this->flux_gauss0.centralize(dx1, dx2);
        result.flux_gauss2 = this->flux_gauss2.centralize(dx1, dx2);
        result.flux_gauss4 = this->flux_gauss4.centralize(dx1, dx2);
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
            model.F.x1,
            model.F.x2,
            model.t.v,
            model.t.g1,
            model.t.g2,
            model.t.x1,
            model.t.x2,
            model.a1.v,
            model.a1.g1,
            model.a1.g2,
            model.a1.x1,
            model.a1.x2,
            model.a2.v,
            model.a2.g1,
            model.a2.g2,
            model.a2.x1,
            model.a2.x2,
            shape[0].v,
            shape[0].g1,
            shape[0].g2,
            shape[0].x1,
            shape[0].x2,
            shape[1].v,
            shape[1].g1,
            shape[1].g2,
            shape[1].x1,
            shape[1].x2,
            model.x1.v,
            model.x1.g1,
            model.x1.g2,
            model.x1.x1,
            model.x1.x2,
            model.x2.v,
            model.x2.g1,
            model.x2.g2,
            model.x2.x1,
            model.x2.x2,
            wdet.v,
            wdet.g1,
            wdet.g2,
            wdet.x1,
            wdet.x2,
            wsel.v,
            wsel.g1,
            wsel.g2,
            wsel.x1,
            wsel.x2,
            mask_value,
            is_peak,
            is_primary,
            flux_gauss0.v,
            flux_gauss0.g1,
            flux_gauss0.g2,
            flux_gauss0.x1,
            flux_gauss0.x2,
            flux_gauss2.v,
            flux_gauss2.g1,
            flux_gauss2.g2,
            flux_gauss2.x1,
            flux_gauss2.x2,
            flux_gauss4.v,
            flux_gauss4.g1,
            flux_gauss4.g2,
            flux_gauss4.x1,
            flux_gauss4.x2,
            flux_gauss0_err,
            flux_gauss2_err,
            flux_gauss4_err,
            fpfs_e1.v,
            fpfs_e1.g1,
            fpfs_e1.g2,
            fpfs_e1.x1,
            fpfs_e1.x2,
            fpfs_e2.v,
            fpfs_e2.g1,
            fpfs_e2.g2,
            fpfs_e2.x1,
            fpfs_e2.x2,
            fpfs_m0.v,
            fpfs_m0.g1,
            fpfs_m0.g2,
            fpfs_m0.x1,
            fpfs_m0.x2,
            fpfs_m2.v,
            fpfs_m2.g1,
            fpfs_m2.g2,
            fpfs_m2.x1,
            fpfs_m2.x2,
            x1_det,
            x2_det,
            block_id
        };
        return row;
    };

    inline void
    from_row(const galRow & row) {
        ra = row.ra;
        dec = row.dec;
        model.F = math::qnumber(
            row.flux,
            row.dflux_dg1, row.dflux_dg2,
            row.dflux_dj1, row.dflux_dj2
        );
        model.t = math::qnumber(
            row.t,
            row.dt_dg1, row.dt_dg2,
            row.dt_dj1, row.dt_dj2
        );
        model.a1 = math::qnumber(
            row.a1,
            row.da1_dg1, row.da1_dg2,
            row.da1_dj1, row.da1_dj2
        );
        model.a2 = math::qnumber(
            row.a2,
            row.da2_dg1, row.da2_dg2,
            row.da2_dj1, row.da2_dj2
        );
        model.x1 = math::qnumber(
            row.x1,
            row.dx1_dg1, row.dx1_dg2,
            row.dx1_dj1, row.dx1_dj2
        );
        model.x2= math::qnumber(
            row.x2,
            row.dx2_dg1, row.dx2_dg2,
            row.dx2_dj1, row.dx2_dj2
        );
        wdet = math::qnumber(
            row.wdet,
            row.dwdet_dg1, row.dwdet_dg2,
            row.dwdet_dj1, row.dwdet_dj2
        );
        wsel = math::qnumber(
            row.wsel,
            row.dwsel_dg1, row.dwsel_dg2,
            row.dwsel_dj1, row.dwsel_dj2
        );
        mask_value = row.mask_value;
        is_peak = row.is_peak;
        is_primary = row.is_primary;
        flux_gauss0 = math::qnumber(
            row.flux_gauss0,
            row.dflux_gauss0_dg1, row.dflux_gauss0_dg2,
            row.dflux_gauss0_dj1, row.dflux_gauss0_dj2
        );
        flux_gauss2 = math::qnumber(
            row.flux_gauss2,
            row.dflux_gauss2_dg1, row.dflux_gauss2_dg2,
            row.dflux_gauss2_dj1, row.dflux_gauss2_dj2
        );
        flux_gauss4 = math::qnumber(
            row.flux_gauss4,
            row.dflux_gauss4_dg1, row.dflux_gauss4_dg2,
            row.dflux_gauss4_dj1, row.dflux_gauss4_dj2
        );
        flux_gauss0_err = row.flux_gauss0_err;
        flux_gauss2_err = row.flux_gauss2_err;
        flux_gauss4_err = row.flux_gauss4_err;
        fpfs_e1 = math::qnumber(
            row.fpfs_e1,
            row.fpfs_de1_dg1, row.fpfs_de1_dg2,
            row.fpfs_de1_dj1, row.fpfs_de1_dj2
        );
        fpfs_e2 = math::qnumber(
            row.fpfs_e2,
            row.fpfs_de2_dg1, row.fpfs_de2_dg2,
            row.fpfs_de2_dj1, row.fpfs_de2_dj2
        );
        fpfs_m0 = math::qnumber(
            row.fpfs_m0,
            row.fpfs_dm0_dg1, row.fpfs_dm0_dg2,
            row.fpfs_dm0_dj1, row.fpfs_dm0_dj2
        );
        fpfs_m2 = math::qnumber(
            row.fpfs_m2,
            row.fpfs_dm2_dg1, row.fpfs_dm2_dg2,
            row.fpfs_dm2_dj1, row.fpfs_dm2_dj2
        );
        x1_det = row.x1_det;
        x2_det = row.x2_det;
        block_id = row.block_id;
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


inline py::array_t<galRow>
make_catalog_empty(
    py::array_t<double> x1,
    py::array_t<double> x2
) {
    if (x1.ndim() != 1 || x2.ndim() != 1) {
        throw std::invalid_argument("x1 and x2 must be one-dimensional arrays");
    }
    if (x1.shape(0) != x2.shape(0)) {
        throw std::invalid_argument("x1 and x2 must have the same length");
    }

    ssize_t n = x1.shape(0);
    auto x1_view = x1.unchecked<1>();
    auto x2_view = x2.unchecked<1>();

    py::array_t<galRow> result(n);
    auto catalog_view = result.mutable_unchecked<1>();

    for (ssize_t i = 0; i < n; ++i) {
        galRow row{};
        row.x1 = x1_view(i);
        row.x2 = x2_view(i);
        row.x1_det = x1_view(i);
        row.x2_det = x2_view(i);
        row.wdet = 1.0;
        row.wsel = 1.0;
        row.is_primary = true;
        catalog_view(i) = row;
    }

    return result;
};


inline std::vector<galNumber>
array_to_objlist(
    const py::array_t<galRow> &records,
    const geometry::block & block
) {
    /* Fast zero‑copy view of the NumPy buffer */
    auto r = records.unchecked<1>();          // one‑dimensional view
    const ssize_t n = r.shape(0);

    std::vector<galNumber> result;
    result.reserve(static_cast<std::size_t>(n));     // upper bound
    double x_min = block.xmin * block.scale;
    double y_min = block.ymin * block.scale;
    double x_max = block.xmax * block.scale;
    double y_max = block.ymax * block.scale;

    for (ssize_t i = 0; i < n; ++i) {
        const galRow &row = r(i);                     // read‑only reference
        if (row.x1_det >= x_min &&
            row.x1_det < x_max &&
            row.x2_det >= y_min &&
            row.x2_det < y_max
        ) {
            galNumber gn;
            gn.from_row(row);
            result.push_back(gn.centralize(block));
        }
    }
    return result;
}


inline std::vector<galNumber>
array_to_objlist(
    const py::array_t<galRow> &records
) {
    /* Fast zero‑copy view of the NumPy buffer */
    auto r = records.unchecked<1>();          // one‑dimensional view
    const ssize_t n = r.shape(0);

    std::vector<galNumber> result;
    result.reserve(static_cast<std::size_t>(n));     // upper bound

    for (ssize_t i = 0; i < n; ++i) {
        const galRow &row = r(i);                     // read‑only reference
        galNumber gn;
        gn.from_row(row);
        result.push_back(gn);
    }
    return result;
}


void pyExportTable(py::module_& m);

} // table
} // anacal
#endif // ANACAL_TABLE_H
