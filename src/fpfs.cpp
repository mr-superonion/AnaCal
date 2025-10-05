#include "anacal.h"

namespace anacal {
void
pyExportFpfs(py::module_& m) {
    py::module_ fpfs = m.def_submodule(
        "fpfs", "submodule for FPFS shear estimation"
    );
    fpfs.attr("fpfs_cut_sigma_ratio") = fpfs::fpfs_cut_sigma_ratio;
    fpfs.attr("fpfs_det_sigma2") = fpfs::fpfs_det_sigma2;
    fpfs::pyExportFpfsBase(fpfs);
    fpfs::pyExportFpfsCatalog(fpfs);
    fpfs::pyExportFpfsImage(fpfs);
}
}
