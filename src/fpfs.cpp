#include "anacal.h"

namespace anacal {
void
pyExportFpfs(py::module_& m) {
    py::module_ fpfs = m.def_submodule(
        "fpfs", "submodule for FPFS shear estimation"
    );
    fpfs::pyExportFpfsBase(fpfs);
    fpfs::pyExportFpfsCatalog(fpfs);
    fpfs::pyExportFpfsForce(fpfs);
    fpfs::pyExportFpfsImage(fpfs);
}
}
