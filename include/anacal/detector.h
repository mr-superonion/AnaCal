#ifndef ANACAL_DETECTOR
#define ANACAL_DETECTOR

#include "image.h"

namespace anacal {
namespace detector {

inline constexpr int drmax = 5;
inline constexpr int drmax2 = drmax * drmax;

inline
void measure_pixel(
    std::vector<table::galNumber> & catalog,
    const std::vector<math::qnumber> & data,
    int x,                                      // index on image
    int y,
    const geometry::block & block,
    double f_min,
    double omega_f,
    double omega_v,
    int drmax_flux,
    int drmax2_flux,
    int drmax_bg,
    int drmax2_bg,
    double nbg,
    double std_noise
) {
    int j = y - block.ymin;
    int i = x - block.xmin;
    int index = j * block.nx + i;
    // ssfunc1(v, omega_v, omega_v) is identically zero for v <= 0, so the
    // product below vanishes unless the centre strictly exceeds every
    // neighbour.  The gate therefore sits exactly on the zero-weight locus:
    // it is a pure short-circuit, never an uncorrected hard cut.
    const double wdet_cut = 0.0;
    // Radius-5 disk: di^2 + dj^2 <= 25 inclusive, excluding the centre, so
    // 80 neighbour differences.  The disk is invariant under a 90 degree
    // rotation, which is what keeps the weight free of additive shear bias.
    math::qnumber wdet = math::qnumber(1.0);
    for (int dj = -drmax; dj <= drmax; dj++) {
        int dj2 = dj * dj;
        for (int di = -drmax; di <= drmax; di++) {
            int dr2 = di * di + dj2;
            if ((dr2 <= drmax2) && (dr2 != 0)) {
                int index2 = (j + dj) * block.nx + (i + di);
                wdet = wdet * math::ssfunc1(
                    data[index] - data[index2],
                    omega_v,
                    omega_v
                );
            }
        }
    }
    if (wdet.v > wdet_cut) {
        table::galNumber src;
        src.x1_det = x * block.scale;
        src.x2_det = y * block.scale;
        src.model.x1.v = src.x1_det;
        src.model.x2.v = src.x2_det;

        math::qnumber fluxbg;
        for (int dj = -drmax_flux; dj <= drmax_flux; dj++) {
            int dj2 = dj * dj;
            for (int di = -drmax_flux; di <= drmax_flux; di++) {
                int dr2 = di * di + dj2;
                if ((dr2 < drmax2_flux) && (dr2 >= drmax2_bg)) {
                    int _i = (j + dj) * block.nx + (i + di);
                    fluxbg = fluxbg + data[_i];
                }
            }
        }

        math::qnumber bkg = fluxbg / nbg;

        src.block_id = block.index;
        // The neighbour-difference product is used directly as the weight.
        // It is already bounded in [0, 1] (every factor is an ssfunc1), so it
        // needs no re-sharpening or renormalisation.
        src.wdet = wdet * math::ssfunc1(
            data[index],
            f_min,
            omega_f
        ) * math::ssfunc1(
            data[index] - bkg * 0.6,
            3.0 * std_noise,
            omega_f
        );
        if (src.wdet.v > 1e-6) catalog.push_back(src);
    }
};

inline std::vector<table::galNumber>
find_peaks_impl(
    const py::array_t<double>& img_array,
    const py::array_t<double>& psf_array,
    double sigma_arcsec,
    double snr_min,
    double variance,
    double omega_f,
    double omega_v,
    const geometry::block & block,
    const std::optional<py::array_t<double>>& noise_array=std::nullopt,
    int image_bound=0
) {
    double sigma_arcsec_det = sigma_arcsec * sqrt2;
    std::vector<math::qnumber> data = prepare_data_block(
        img_array,
        psf_array,
        sigma_arcsec_det,
        block,
        noise_array
    );

    double std_noise = std::pow(
        get_smoothed_variance(
            block.scale,
            sigma_arcsec_det,
            psf_array,
            variance
        ), 0.5
    );
    // Secondary peak cut
    double f_min = std_noise * snr_min;
    double f_cut = f_min - omega_f;
    // ssfunc1(v, omega_v, omega_v) vanishes for v <= 0, so this cheap
    // pre-filter sits exactly where the weight is already zero.
    const double v_cut = 0.0;

    int image_ny = img_array.shape(0);
    int image_nx = img_array.shape(1);

    // fluxdet is for 0 to 2 arcsec
    int drmax_flux = static_cast<int>(2.0 / block.scale) + 1;
    int drmax2_flux = drmax_flux * drmax_flux;
    // background is for 1 arcsec to 2 arcsec
    int drmax_bg = static_cast<int>(1.0 / block.scale) + 1;
    int drmax2_bg = drmax_bg * drmax_bg;

    double nbg = 0.0;
    for (int dj = -drmax_flux; dj <= drmax_flux; ++dj) {
        int dj2 = dj * dj;
        for (int di = -drmax_flux; di <= drmax_flux; ++di) {
            int dr2 = di * di + dj2;
            if (dr2 >= drmax2_bg && dr2 < drmax2_flux) {
                nbg = nbg + 1;
            }
        }
    }

    int ystart = std::max(image_bound, block.ymin_in);
    int yend = std::min(image_ny - image_bound, block.ymax_in);
    int xstart = std::max(image_bound, block.xmin_in);
    int xend = std::min(image_nx - image_bound, block.xmax_in);

    std::vector<table::galNumber> catalog;
    for (int y = ystart; y < yend; ++y) {
        int j = y - block.ymin;
        for (int x = xstart; x < xend; ++x) {
            int i = x - block.xmin;
            // data index
            int index = j * block.nx + i;
            if (
                (data[index].v > f_cut) &&
                (data[index].v - data[j * block.nx + (i + 1)].v > v_cut) &&
                (data[index].v - data[j * block.nx + (i - 1)].v > v_cut) &&
                (data[index].v - data[(j + 1) * block.nx + i].v > v_cut) &&
                (data[index].v - data[(j - 1) * block.nx + i].v > v_cut)
            ) {
                measure_pixel(
                    catalog,
                    data,
                    x,
                    y,
                    block,
                    f_min,
                    omega_f,
                    omega_v,
                    drmax_flux,
                    drmax2_flux,
                    drmax_bg,
                    drmax2_bg,
                    nbg,
                    std_noise
                );
            }

        }
    }
    return catalog;
};

inline std::vector<table::galNumber>
find_peaks(
    const py::array_t<double>& img_array,
    const py::array_t<double>& psf_array,
    double sigma_arcsec,
    double snr_min,
    double variance,
    double omega_f,
    double omega_v,
    const geometry::block & block,
    const std::optional<py::array_t<double>>& noise_array=std::nullopt,
    int image_bound=0
) {
    std::vector<table::galNumber> cat = find_peaks_impl(
        img_array,
        psf_array,
        sigma_arcsec,
        snr_min,
        variance,
        omega_f,
        omega_v,
        block,
        noise_array,
        image_bound
    );

    // --------------------------------------------------------------------
    // DISABLED: neighbour-competition ("deblend") re-weighting.
    //
    // ``ss`` averaged the wdet of *other* detected peaks inside r^2 < 8
    // (r < 2.83 pix).  Step 1 makes that region provably empty: if a pixel
    // is detected then it strictly exceeds every neighbour inside the
    // radius-``drmax`` stencil, and because the stencil is symmetric each of
    // those neighbours picks up an ssfunc1 with a negative argument and so
    // gets wdet = 0.  Detected peaks are therefore always more than
    // ``drmax`` pixels apart, which exceeds the 2.83 pix footprint for any
    // drmax >= 3, so ``ss`` is identically zero for every source.  Measured
    // directly on a dense field: minimum peak separation was 5.0 pix at
    // drmax = 4, with zero pairs inside the footprint.
    //
    // With ss == 0 the factor below degenerates into ssfunc1(wdet, 0.4,
    // 0.399): a pure re-sharpening of wdet, the same construction as the
    // p_min/omega_p layer that was removed, but wider (zero below wdet =
    // 0.001, one above 0.799).  It performs no deblending, and because its
    // thresholds are fixed while wdet's scale depends on the stencil size,
    // it silently couples the two.  Kept here for reference only.
    //
    // std::vector<math::qnumber> data(block.nx * block.ny);
    // for (const table::galNumber & src: cat){
    //     const ngmix::NgmixGaussian & model = src.model;
    //     int i = static_cast<int>(
    //         std::round(model.x1.v / block.scale)
    //     ) - block.xmin;
    //     int j = static_cast<int>(
    //         std::round(model.x2.v / block.scale)
    //     ) - block.ymin;
    //     data[j * block.nx + i] = src.wdet;
    // }
    // for (table::galNumber & src: cat){
    //     const ngmix::NgmixGaussian & model = src.model;
    //     int i = static_cast<int>(
    //         std::round(model.x1.v / block.scale)
    //     ) - block.xmin;
    //     int j = static_cast<int>(
    //         std::round(model.x2.v / block.scale)
    //     ) - block.ymin;
    //     math::qnumber ss;
    //     int nss = 0;
    //     for (int jj = j - 3; jj <= j + 3; ++jj) {
    //         int dy = jj - j;
    //         for (int ii = i - 3; ii <= i + 3; ++ii) {
    //             int dx = ii -i;
    //             // radius
    //             int r2 = dx * dx + dy * dy;
    //             if ((r2 < 8) && (r2!=0)) {
    //                 ss = ss + data[jj * block.nx + ii];
    //                 nss = nss + 1;
    //             }
    //         }
    //     }
    //     // average over the footprint rather than the bare sum
    //     if (nss > 0) ss = ss / static_cast<double>(nss);
    //     src.wdet = math::ssfunc1(
    //         src.wdet - ss,
    //         0.4,
    //         0.399
    //     );
    // }
    // --------------------------------------------------------------------

    std::vector<table::galNumber> catalog;
    catalog.reserve(cat.size());
    for (table::galNumber & src: cat){
        if (src.wdet.v > 1e-8) catalog.push_back(src);
    }
    return catalog;
};

void pyExportDetector(py::module_& m);

} // detector
} // anacal

#endif // ANACAL_DETECTOR
