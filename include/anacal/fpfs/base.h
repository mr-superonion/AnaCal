#ifndef ANACAL_FPFS_BASE_H
#define ANACAL_FPFS_BASE_H

#include "../image.h"
#include "../psf.h"
#include "../math.h"
#include "../mask.h"
#include "../stdafx.h"

namespace anacal {
    inline constexpr double fpfs_cut_sigma_ratio = 1.6;
    inline constexpr double fpfs_det_sigma2 = 0.04;
    inline constexpr double fpfs_pnr = 0.8;
}

#endif // ANACAL_FPFS_BASE_H
