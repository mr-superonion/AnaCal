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
    // Local background level under the source, as estimated by the detector's
    // ring cascade (detector.h).  Same units as the smoothed image, i.e. on the
    // catalog's mag_zero.
    double bkg;
    double dbkg_dg1;
    double dbkg_dg2;
    double dbkg_dj1;
    double dbkg_dj2;
    double wsel;
    double dwsel_dg1;
    double dwsel_dg2;
    double dwsel_dj1;
    double dwsel_dj2;
    float n_mask_base;
    float n_mask_discontinuity;
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
    double flux_gauss0_err;
    double flux_gauss2_err;
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
};

struct galNumber {
    // value with derivatives to Gaussian model parameters
    ngmix::NgmixGaussian model;
    // Fail-closed: both weights start at zero and are only given real
    // values by detection (wdet, wsel) and measurement (wsel), so a source
    // that never passes through those stages stays inert in every weighted
    // sum instead of counting as selected.  Detection catalogs passed into
    // the measurement are expected to come from AnaCal detection, whose
    // sources all carry wdet > wdet_min; the measurement computes
    // wsel = wdet * (FPFS cut), so a source entering with wdet = 0 simply
    // stays at zero weight.
    math::qnumber wdet = math::qnumber(0.0);
    // local background under the source; set by detector::measure_pixel
    math::qnumber bkg = math::qnumber(0.0);
    math::qnumber wsel = math::qnumber(0.0);
    // Gaussian-weighted MEAN of mask bit 0 (masked pixels) over the
    // source footprint: a FRACTION in [0, 1], not a density -- see
    // mask::add_mask_fraction_columns. Sources above
    // ``n_mask_base_max`` are skipped by the measurement.
    float n_mask_base=0.0f;
    // The same fraction for mask bit 1 (DISCONTINUITY: chip gaps,
    // clipped/rejected coadd inputs -- real data, wrong CoaddPsf).
    // Never used for skipping; carried as a catalog column only.
    float n_mask_discontinuity=0.0f;
    bool is_primary=true;
    bool initialized=false;
    math::lossNumber loss;
    math::qnumber fpfs_e1;
    math::qnumber fpfs_e2;
    math::qnumber fpfs_m0;
    math::qnumber fpfs_m2;
    math::qnumber flux_gauss0 = math::qnumber(0.0);
    math::qnumber flux_gauss2 = math::qnumber(0.0);
    double flux_gauss0_err = 0.0;
    double flux_gauss2_err = 0.0;
    double ra = 0.0;
    double dec = 0.0;
    double x1_det, x2_det;
    // Which cell owns this source.  INTERNAL: assigned from x1_det/x2_det
    // by Task::assign_cell_ids (or detector::measure_pixel at detection);
    // not an output column.
    int cell_id = -1;

    galNumber() = default;

    galNumber(
        ngmix::NgmixGaussian model,
        math::qnumber wdet,
        float n_mask_base,
        math::lossNumber loss
    ) : model(model), wdet(wdet),
        n_mask_base(n_mask_base), loss(loss) {}

    // In-place: a galNumber is 1.2 kB but only the g1/g2 slots of ten
    // qnumbers actually change, so mutating beats copying the struct.
    // One signed implementation: a POSITIVE (dx1, dx2) -- the source
    // position wrt the cell center -- shifts the reference point to
    // the CELL CENTER (centralize, before per-cell fitting); the
    // NEGATED shift moves it back to the DETECTION PEAK (decentralize,
    // after fitting).
    inline void
    shift_reference(double dx1, double dx2) {
        this->wdet.shift_reference(dx1, dx2);
        this->bkg.shift_reference(dx1, dx2);
        this->wsel.shift_reference(dx1, dx2);
        this->model.shift_reference(dx1, dx2);
        this->fpfs_e1.shift_reference(dx1, dx2);
        this->fpfs_e2.shift_reference(dx1, dx2);
        this->fpfs_m0.shift_reference(dx1, dx2);
        this->fpfs_m2.shift_reference(dx1, dx2);
        this->flux_gauss0.shift_reference(dx1, dx2);
        this->flux_gauss2.shift_reference(dx1, dx2);
    };

    inline void
    centralize(const geometry::cell & cell) {
        double dx1 = this->x1_det - cell.xcen * cell.scale;
        double dx2 = this->x2_det - cell.ycen * cell.scale;
        this->shift_reference(dx1, dx2);
    };

    inline void
    decentralize(const geometry::cell & cell) {
        double dx1 = this->x1_det - cell.xcen * cell.scale;
        double dx2 = this->x2_det - cell.ycen * cell.scale;
        this->shift_reference(-dx1, -dx2);
    };

// One qnumber spans five catalog columns (value, _dg1, _dg2, _dj1, _dj2).
// The derivative stem is a separate macro argument because the column naming
// is not uniform (flux -> dflux_dg1, but fpfs_e1 -> fpfs_de1_dg1).  By-name
// assignment makes a wrong or misplaced name a compile error -- unlike the
// positional aggregate this replaces, where one misplaced entry silently
// shifted every later column.  tests/test_table.py round-trips every column
// through from_row/to_row.
#define ANACAL_ROW_PUT_Q(val_name, d_stem, qn) \
    row.val_name = (qn).v; \
    row.d_stem##_dg1 = (qn).g1; \
    row.d_stem##_dg2 = (qn).g2; \
    row.d_stem##_dj1 = (qn).x1; \
    row.d_stem##_dj2 = (qn).x2

#define ANACAL_ROW_GET_Q(val_name, d_stem, target) \
    target = math::qnumber( \
        row.val_name, \
        row.d_stem##_dg1, row.d_stem##_dg2, \
        row.d_stem##_dj1, row.d_stem##_dj2 \
    )

    inline galRow
    to_row() const {
        std::array<math::qnumber, 2> shape = model.get_shape();
        galRow row{};
        row.ra = ra;
        row.dec = dec;
        ANACAL_ROW_PUT_Q(flux, dflux, model.F);
        ANACAL_ROW_PUT_Q(t, dt, model.t);
        ANACAL_ROW_PUT_Q(a1, da1, model.a1);
        ANACAL_ROW_PUT_Q(a2, da2, model.a2);
        ANACAL_ROW_PUT_Q(e1, de1, shape[0]);
        ANACAL_ROW_PUT_Q(e2, de2, shape[1]);
        ANACAL_ROW_PUT_Q(x1, dx1, model.x1);
        ANACAL_ROW_PUT_Q(x2, dx2, model.x2);
        ANACAL_ROW_PUT_Q(wdet, dwdet, wdet);
        ANACAL_ROW_PUT_Q(bkg, dbkg, bkg);
        ANACAL_ROW_PUT_Q(wsel, dwsel, wsel);
        row.n_mask_base = n_mask_base;
        row.n_mask_discontinuity = n_mask_discontinuity;
        row.is_primary = is_primary;
        ANACAL_ROW_PUT_Q(flux_gauss0, dflux_gauss0, flux_gauss0);
        ANACAL_ROW_PUT_Q(flux_gauss2, dflux_gauss2, flux_gauss2);
        row.flux_gauss0_err = flux_gauss0_err;
        row.flux_gauss2_err = flux_gauss2_err;
        ANACAL_ROW_PUT_Q(fpfs_e1, fpfs_de1, fpfs_e1);
        ANACAL_ROW_PUT_Q(fpfs_e2, fpfs_de2, fpfs_e2);
        ANACAL_ROW_PUT_Q(fpfs_m0, fpfs_dm0, fpfs_m0);
        ANACAL_ROW_PUT_Q(fpfs_m2, fpfs_dm2, fpfs_m2);
        row.x1_det = x1_det;
        row.x2_det = x2_det;
        return row;
    };

    inline void
    from_row(const galRow & row) {
        ra = row.ra;
        dec = row.dec;
        ANACAL_ROW_GET_Q(flux, dflux, model.F);
        ANACAL_ROW_GET_Q(t, dt, model.t);
        ANACAL_ROW_GET_Q(a1, da1, model.a1);
        ANACAL_ROW_GET_Q(a2, da2, model.a2);
        // e1/e2 are DERIVED columns: to_row computes them from a1/a2/t via
        // model.get_shape(), so there is nothing to restore for them here.
        ANACAL_ROW_GET_Q(x1, dx1, model.x1);
        ANACAL_ROW_GET_Q(x2, dx2, model.x2);
        ANACAL_ROW_GET_Q(wdet, dwdet, wdet);
        ANACAL_ROW_GET_Q(bkg, dbkg, bkg);
        ANACAL_ROW_GET_Q(wsel, dwsel, wsel);
        n_mask_base = row.n_mask_base;
        n_mask_discontinuity = row.n_mask_discontinuity;
        is_primary = row.is_primary;
        ANACAL_ROW_GET_Q(flux_gauss0, dflux_gauss0, flux_gauss0);
        ANACAL_ROW_GET_Q(flux_gauss2, dflux_gauss2, flux_gauss2);
        flux_gauss0_err = row.flux_gauss0_err;
        flux_gauss2_err = row.flux_gauss2_err;
        ANACAL_ROW_GET_Q(fpfs_e1, fpfs_de1, fpfs_e1);
        ANACAL_ROW_GET_Q(fpfs_e2, fpfs_de2, fpfs_e2);
        ANACAL_ROW_GET_Q(fpfs_m0, fpfs_dm0, fpfs_m0);
        ANACAL_ROW_GET_Q(fpfs_m2, fpfs_dm2, fpfs_m2);
        x1_det = row.x1_det;
        x2_det = row.x2_det;
    };

#undef ANACAL_ROW_PUT_Q
#undef ANACAL_ROW_GET_Q
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
        // Weights stay 0 at initialization (galRow{} zero-fills them): an
        // unmeasured table carries no weight.  To use this catalog as a
        // forced ``detection`` input the caller must set wdet explicitly
        // (e.g. catalog["wdet"] = 1.0); the measurement computes
        // wsel = wdet * (FPFS cut), so sources left at wdet = 0 come back
        // with zero selection weight.
        row.is_primary = true;
        catalog_view(i) = row;
    }

    return result;
};


inline std::vector<galNumber>
array_to_objlist(
    const py::array_t<galRow> &records,
    const geometry::cell & cell
) {
    /* Fast zero‑copy view of the NumPy buffer */
    auto r = records.unchecked<1>();          // one‑dimensional view
    const ssize_t n = r.shape(0);

    std::vector<galNumber> result;
    result.reserve(static_cast<std::size_t>(n));     // upper bound
    double x_min = cell.xmin * cell.scale;
    double y_min = cell.ymin * cell.scale;
    double x_max = cell.xmax * cell.scale;
    double y_max = cell.ymax * cell.scale;

    for (ssize_t i = 0; i < n; ++i) {
        const galRow &row = r(i);                     // read‑only reference
        if (row.x1_det >= x_min &&
            row.x1_det < x_max &&
            row.x2_det >= y_min &&
            row.x2_det < y_max
        ) {
            galNumber gn;
            gn.from_row(row);
            gn.centralize(cell);
            result.push_back(gn);
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
