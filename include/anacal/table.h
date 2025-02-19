#ifndef ANACAL_TABLE_H
#define ANACAL_TABLE_H

#include "stdafx.h"
#include "ngmix.h"


namespace anacal {
namespace table {

struct block {
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

    block() = default;
    block(
        int xc, int yc, int xmi, int ymi, int xma, int yma,
        int xmi_in, int ymi_in, int xma_in, int yma_in
    ) : xcen(xc), ycen(yc), xmin(xmi), ymin(ymi), xmax(xma), ymax(yma),
        xmin_in(xmi_in), ymin_in(ymi_in), xmax_in(xma_in), ymax_in(yma_in) {}
};

inline std::vector<block> get_block_center_list(
    int img_nx,
    int img_ny,
    int block_nx,
    int block_ny,
    int block_overlap
) {

    int block_ny2 = block_ny / 2;
    int block_nx2 = block_nx / 2;
    // Determine number of patches
    // y direction
    int npatch_y = img_ny / (block_ny - block_overlap);
    float npatch_y_f = img_ny / static_cast<float>(block_ny - block_overlap);
    if (npatch_y_f > npatch_y) {
        npatch_y = npatch_y + 1;
    }
    int nyy = npatch_y * (block_ny - block_overlap) + block_overlap;
    int npad_y = (nyy - img_ny) / 2;

    // x direction
    int npatch_x = img_nx / (block_nx - block_overlap);
    float npatch_x_f = img_nx / static_cast<float>(block_nx - block_overlap);
    if (npatch_x_f > npatch_x) {
        npatch_x = npatch_x + 1;
    }
    int nxx = npatch_x * (block_nx - block_overlap) + block_overlap;
    int npad_x = (nxx - img_nx) / 2;

    int block_bound = std::max(block_overlap / 2, 3);

    std::vector<block> result(npatch_y * npatch_x);
    // Do detection in each patch
    for (int j = 0; j < npatch_y; ++j) {
        int ycen = (block_ny - block_overlap) * j + block_ny2 - npad_y;
        int ymin = ycen - block_ny2; // (starting point)
        int ymax = ycen + block_ny2; // (end point not included)
        int ymin_in = ymin + block_bound;
        int ymax_in = ymax - block_bound;
        for (int i = 0; i < npatch_x; ++i) {
            int xcen = (block_nx - block_overlap) * i + block_nx2 - npad_x;
            int index = j * npatch_x + i;
            int xmin = xcen - block_nx2;
            int xmax = xcen + block_nx2;
            int xmin_in = xmin + block_bound;
            int xmax_in = xmax - block_bound;
            result[index] = block(
                xcen,
                ycen,
                xmin,
                ymin,
                xmax,
                ymax,
                xmin_in,
                ymin_in,
                xmax_in,
                ymax_in
            );
        }
    }
    return result;
};

struct galNumber {
    // value with derivatives to Gaussian model parameters
    ngmix::modelNumber params;
    math::qnumber wdet;
    int mask_value=0;
    ngmix::lossNumber loss;

    galNumber() = default;

    galNumber(
        ngmix::modelNumber params,
        math::qnumber wdet,
        int mask_value,
        ngmix::lossNumber loss
    ) : params(params), wdet(wdet), mask_value(mask_value), loss(loss) {}
};

void pyExportTable(py::module_& m);

} // table
} // anacal
#endif // ANACAL_TABLE_H
