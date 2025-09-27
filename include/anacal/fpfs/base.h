#ifndef ANACAL_FPFS_BASE_H
#define ANACAL_FPFS_BASE_H

#include "../image.h"
#include "../math.h"
#include "../mask.h"
#include "../stdafx.h"

namespace anacal {
    inline constexpr double fpfs_det_sigma2 = 0.04;
    inline constexpr double fpfs_cut_sigma_ratio = 1.6;

    namespace fpfs_detail {
        inline constexpr int idx4(int n, int m, int norder) noexcept {
            static_cast<void>(norder);
            return n * 4 + m;
        }
    }
}

#endif // ANACAL_FPFS_BASE_H
