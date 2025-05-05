#ifndef ANACAL_DETECTOR
#define ANACAL_DETECTOR

#include "image.h"
#include "math.h"
#include "stdafx.h"
#include "table.h"

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
    double v_min,
    double omega_v,
    double p_min,
    double omega_p,
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
    double wdet_cut = p_min - omega_p;
    // pixel value greater than threshold
    math::qnumber wdet = math::qnumber(1.0);
    for (int dj = -drmax; dj <= drmax; dj++) {
        int dj2 = dj * dj;
        for (int di = -drmax; di <= drmax; di++) {
            int dr2 = di * di + dj2;
            if ((dr2 <= drmax2) && (dr2 != 0)) {
                int index2 = (j + dj) * block.nx + (i + di);
                wdet = wdet * math::ssfunc1(
                    data[index] - data[index2],
                    v_min,
                    omega_v
                );
            }
        }
    }
    if (wdet.v > wdet_cut) {
        int id1 = j * block.nx + (i + 1);
        int id2 = j * block.nx + (i - 1);
        int id3 = (j + 1) * block.nx + i;
        int id4 = (j - 1) * block.nx + i;
        table::galNumber src;
        src.model.x1.v = x * block.scale;
        src.model.x2.v = y * block.scale;
        src.is_peak = (
            (data[index].v > data[id1].v) &&
            (data[index].v > data[id2].v) &&
            (data[index].v > data[id3].v) &&
            (data[index].v > data[id4].v)
        );

        math::qnumber fluxbg, fluxap2;
        for (int dj = -drmax_flux; dj <= drmax_flux; dj++) {
            int dj2 = dj * dj;
            for (int di = -drmax_flux; di <= drmax_flux; di++) {
                int dr2 = di * di + dj2;
                if ((dr2 < drmax2_flux)) {
                    int _i = (j + dj) * block.nx + (i + di);
                    fluxap2 = fluxap2 + data[_i];
                    if ((dr2 >= drmax2_bg)) {
                        fluxbg = fluxbg + data[_i];
                    }
                }
            }
        }
        src.peakv = data[index];
        src.bkg = fluxbg / nbg;
        src.fluxap2 = fluxap2;
        src.wsel = math::ssfunc1(
            data[index],
            f_min,
            omega_f
        )* math::ssfunc1(
            data[index] - src.bkg * 0.9,
            5.0 * std_noise,
            omega_f
        );
        src.block_id = block.index;
        src.wdet = math::ssfunc1(
            wdet,
            p_min,
            omega_p
        ) * math::ssfunc1(
            data[index],
            f_min,
            omega_f
        )* math::ssfunc1(
            data[index] - src.bkg * 0.9,
            5.0 * std_noise,
            omega_f
        );
        if (src.wdet.v > 1e-8) catalog.push_back(src);
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
    double v_min,
    double omega_v,
    double p_min,
    double omega_p,
    const geometry::block & block,
    const std::optional<py::array_t<double>>& noise_array=std::nullopt,
    int image_bound=0
) {
    double sigma_arcsec_det = sigma_arcsec * 1.414;
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
    double v_cut = v_min - omega_v;

    int image_ny = img_array.shape(0);
    int image_nx = img_array.shape(1);

    // fluxdet is for 0 to 2 arcsec
    int drmax_flux = static_cast<int>(2.0 / block.scale) + 1;
    int drmax2_flux = drmax_flux * drmax_flux;
    // background is for 1 arcsec to 2 arcsec
    int drmax_bg = static_cast<int>(1.0 / block.scale) + 1;
    int drmax2_bg = drmax_bg * drmax_bg;

    double nbg=0.0;
    for (int dj = -drmax_flux; dj <= drmax_flux; ++dj) {
        int dj2 = dj * dj;
        for (int di = -drmax_flux; di <= drmax_flux; ++di) {
            int dr2 = di * di + dj2;
            if (dr2 >= drmax2_bg && dr2 < drmax2_flux) {
                nbg=nbg+1;
            }
        }
    }

    int ystart = std::max(image_bound, block.ymin_in);
    int yend = std::min(image_ny - image_bound, block.ymax_in);
    int xstart = std::max(image_bound, block.xmin_in);
    int xend = std::min(image_nx - image_bound, block.xmax_in);

    std::vector<table::galNumber> catalog;
    catalog.reserve((yend - ystart) * (xend - xstart) / 2500);
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
                    v_min,
                    omega_v,
                    p_min,
                    omega_p,
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
    double v_min,
    double omega_v,
    double p_min,
    double omega_p,
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
        v_min,
        omega_v,
        p_min,
        omega_p,
        block,
        noise_array,
        image_bound
    );
    std::vector<math::qnumber> data(block.nx * block.ny);
    for (const table::galNumber & src: cat){
        const ngmix::NgmixGaussian & model = src.model;
        int i = static_cast<int>(
            std::round(model.x1.v / block.scale)
        ) - block.xmin;
        int j = static_cast<int>(
            std::round(model.x2.v / block.scale)
        ) - block.ymin;
        data[j * block.nx + i] = src.wdet;
    }
    std::vector<table::galNumber> catalog;
    catalog.reserve(cat.size() / 2);
    for (table::galNumber & src: cat){
        const ngmix::NgmixGaussian & model = src.model;
        int i = static_cast<int>(
            std::round(model.x1.v / block.scale)
        ) - block.xmin;
        int j = static_cast<int>(
            std::round(model.x2.v / block.scale)
        ) - block.ymin;
        math::qnumber ss;
        for (int jj = j - 3; jj <= j + 3; ++jj) {
            int dy = jj - j;
            for (int ii = i - 3; ii <= i + 3; ++ii) {
                int dx = ii -i;
                // radius
                int r2 = dx * dx + dy * dy;
                if ((r2 < 8) && (r2!=0)) {
                    ss = ss + data[jj * block.nx + ii];
                }
            }
        }
        src.wdet = math::ssfunc1(
            src.wdet - ss,
            0.4,
            0.399
        );
        src.model.F = src.fluxap2;
        if (src.wdet.v > 1e-8) catalog.push_back(src);
    }
    return catalog;
};

void pyExportDetector(py::module_& m);

} // detector
} // anacal

#endif // ANACAL_DETECTOR
