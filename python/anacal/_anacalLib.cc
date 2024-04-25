#include "anacal.h"


namespace anacal {
    PYBIND11_MODULE(_anacal, m)
    {
        pyExportModel(m);
        pyExportImage(m);
        pyExportFpfs(m);
        pyExportNoise(m);
        pyExportPsf(m);
        pyExportMask(m);
    }
}
