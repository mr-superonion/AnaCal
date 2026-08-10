#ifndef ANACAL_GEO_H
#define ANACAL_GEO_H

#include "stdafx.h"

namespace anacal {
namespace geometry {

struct cell {
    int xcen = 0;
    int ycen = 0;
    int xmin = 0;
    int ymin = 0;
    int xmax = 0;
    int ymax = 0;
    int xmin_in = 0;
    int ymin_in = 0;
    int xmax_in = 0;
    int ymax_in = 0;
    int nx = 0;
    int ny = 0;
    double scale = 0.2;
    int index = 0;

    std::vector<double> xvs;
    std::vector<double> yvs;
    std::vector<bool> xmsk;
    std::vector<bool> ymsk;
    py::array_t<double> psf_array;

    cell() = default;
    cell(
        int xc, int yc, int xmi, int ymi, int xma, int yma,
        int xmi_in, int ymi_in, int xma_in, int yma_in, double scale,
        int index
    ) : xcen(xc), ycen(yc), xmin(xmi), ymin(ymi), xmax(xma), ymax(yma),
        xmin_in(xmi_in), ymin_in(ymi_in), xmax_in(xma_in), ymax_in(yma_in),
        scale(scale), index(index) {
        this->nx = xmax - xmin;
        this->ny = ymax - ymin;
        this->xvs.resize(this->nx);
        this->yvs.resize(this->ny);
        for (int i = 0; i < this->nx; ++i) {
            this->xvs[i] = (i + this->xmin) * this->scale;
        }
        for (int i = 0; i < this->ny; ++i) {
            this->yvs[i] = (i + this->ymin) * this->scale;
        }
        this->xmsk.assign(this->nx, true);
        this->ymsk.assign(this->ny, true);
    }
};

inline std::vector<cell> get_cell_list(
    int img_nx,
    int img_ny,
    int cell_nx,
    int cell_ny,
    int cell_overlap,
    double scale
) {
    if ((cell_overlap % 2 != 0) || (cell_overlap < 0)) {
        throw std::runtime_error(
            "Cell Error: cell_overlap is not an even number"
        );
    }
    int cell_ny2 = cell_ny / 2;
    int cell_nx2 = cell_nx / 2;
    // Determine number of patches
    // y direction
    int npatch_y = img_ny / (cell_ny - cell_overlap);
    float npatch_y_f = img_ny / static_cast<float>(cell_ny - cell_overlap);
    if (npatch_y_f > npatch_y) {
        npatch_y = npatch_y + 1;
    }
    int nyy = npatch_y * (cell_ny - cell_overlap) + cell_overlap;
    int npad_y = (nyy - img_ny) / 2;

    // x direction
    int npatch_x = img_nx / (cell_nx - cell_overlap);
    float npatch_x_f = img_nx / static_cast<float>(cell_nx - cell_overlap);
    if (npatch_x_f > npatch_x) {
        npatch_x = npatch_x + 1;
    }
    int nxx = npatch_x * (cell_nx - cell_overlap) + cell_overlap;
    int npad_x = (nxx - img_nx) / 2;

    int cell_bound = std::max(cell_overlap / 2, 3);

    std::vector<cell> result(npatch_y * npatch_x);
    // Do detection in each patch
    for (int j = 0; j < npatch_y; ++j) {
        int ycen = (cell_ny - cell_overlap) * j + cell_ny2 - npad_y;
        int ymin = ycen - cell_ny2; // (starting point)
        int ymax = ycen + cell_ny2; // (end point not included)
        int ymin_in = ymin + cell_bound;
        int ymax_in = ymax - cell_bound;
        for (int i = 0; i < npatch_x; ++i) {
            int xcen = (cell_nx - cell_overlap) * i + cell_nx2 - npad_x;
            int index = j * npatch_x + i;
            int xmin = xcen - cell_nx2;
            int xmax = xcen + cell_nx2;
            int xmin_in = xmin + cell_bound;
            int xmax_in = xmax - cell_bound;
            result[index] = cell(
                xcen,
                ycen,
                xmin,
                ymin,
                xmax,
                ymax,
                xmin_in,
                ymin_in,
                xmax_in,
                ymax_in,
                scale,
                index
            );
            cell & bb = result[index];
            // The image-pixel index of column i is i + xmin by construction
            // (xvs[i] = (i + xmin) * scale).  Recovering it as xvs[i] / scale
            // truncates toward zero, so index -1 comes back as -0.999... -> 0
            // and one column of padding outside the image is marked valid.
            for (int i = 0; i < bb.nx; ++i) {
                int ii = i + bb.xmin;
                bb.xmsk[i] = (ii >= 0) && (ii < img_nx);
            }
            for (int i = 0; i < bb.ny; ++i) {
                int ii = i + bb.ymin;
                bb.ymsk[i] = (ii >= 0) && (ii < img_ny);
            }

        }
    }
    return result;
};

void pyExportGeometry(py::module_& m);

} // geometry
} // anacal
#endif // ANACAL_GEO_H
