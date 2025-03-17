#ifndef ANACAL_DETECTOR
#define ANACAL_DETECTOR

#include "image.h"
#include "math.h"
#include "stdafx.h"
#include "table.h"

namespace anacal {
namespace detector {

inline int drmax2 = 16;

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
    double omega_p
) {
    int j = y - block.ymin;
    int i = x - block.xmin;
    int index = j * block.nx + i;
    int id1 = j * block.nx + (i + 1);
    int id2 = j * block.nx + (i - 1);
    int id3 = (j + 1) * block.nx + i;
    int id4 = (j - 1) * block.nx + i;
    double wdet_cut = p_min - omega_p;
    // pixel value greater than threshold
    math::qnumber wdet = math::qnumber(1.0);
    for (int dj = -1; dj <= 1; dj++) {
        int dj2 = dj * dj;
        for (int di = -1; di <= 1; di++) {
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
        table::galNumber src;
        src.model.x1.v = x * block.scale;
        src.model.x2.v = y * block.scale;
        src.is_peak = (
            (data[index].v > data[id1].v) &&
            (data[index].v > data[id2].v) &&
            (data[index].v > data[id3].v) &&
            (data[index].v > data[id4].v)
        );

        int drmax_flux = static_cast<int>(2.0 / block.scale) + 1;
        int drmax2_flux = drmax_flux * drmax_flux;
        math::qnumber fluxdet;
        for (int dj = -drmax_flux; dj <= drmax_flux; dj++) {
            int dj2 = dj * dj;
            for (int di = -drmax_flux; di <= drmax_flux; di++) {
                int dr2 = di * di + dj2;
                if (dr2 < drmax2_flux) {
                    fluxdet = (
                        fluxdet
                        + data[(j + dj) * block.nx + (i + di)]
                    );
                }
            }
        }
        src.fluxdet = fluxdet;
        src.wdet = math::ssfunc1(
            wdet,
            p_min,
            omega_p
        ) * math::ssfunc1(
            data[index],
            f_min,
            omega_f
        );
        src.model.F = fluxdet;
        catalog.push_back(src);
    }
};

inline std::vector<table::galNumber>
find_peaks(
    const py::array_t<double>& img_array,
    const py::array_t<double>& psf_array,
    double sigma_arcsec,
    double f_min,
    double omega_f,
    double v_min,
    double omega_v,
    double p_min,
    double omega_p,
    const geometry::block & block,
    const std::optional<py::array_t<double>>& noise_array=std::nullopt,
    int image_bound=0
) {
    std::vector<math::qnumber> data = prepare_data_block(
        img_array,
        psf_array,
        sigma_arcsec,
        block,
        noise_array
    );

    // Secondary peak cut
    double f_cut = f_min - omega_f;
    double v_cut = v_min - omega_v;

    int image_ny = img_array.shape(0);
    int image_nx = img_array.shape(1);

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
                    v_min,
                    omega_v,
                    p_min,
                    omega_p
                );
            }

        }
    }
    return catalog;
}

void pyExportDetector(py::module_& m);

} // detector
} // anacal

#endif // ANACAL_DETECTOR
