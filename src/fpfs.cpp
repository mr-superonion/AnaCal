#include "anacal.h"

namespace anacal {
void
pyExportFpfs(py::module_& m) {
    py::module_ fpfs = m.def_submodule(
        "fpfs", "submodule for FPFS shear estimation"
    );
    fpfs.attr("fpfs_cut_sigma_ratio") = fpfs_cut_sigma_ratio;
    fpfs.attr("fpfs_det_sigma2") = fpfs_det_sigma2;
    fpfs.attr("fpfs_pnr") = fpfs_pnr;
    pybindFpfsCatalog(fpfs);
    pybindFpfsImage(fpfs);
}
}
