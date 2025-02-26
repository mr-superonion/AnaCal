#include "anacal.h"


namespace anacal {
    PYBIND11_MODULE(_anacal, m)
    {
        pyExportModel(m);
        pyExportImage(m);
        pyExportFpfs(m);
        math::pyExportMath(m);
        noise::pyExportNoise(m);
        psf::pyExportPsf(m);
        mask::pyExportMask(m);
        ngmix::pyExportNgmix(m);
        table::pyExportTable(m);
        detector::pyExportDetector(m);
        geometry::pyExportGeometry(m);
        task::pyExportTask(m);
    }
}
