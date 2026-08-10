#include "anacal.h"


namespace anacal {
    PYBIND11_MODULE(_anacal, m)
    {
        // Thread safety note: the measurement runners release the GIL
        // around their compute, so FFTW planning and execution can
        // overlap between Python threads.  anacal serializes the FFTW
        // planner itself with a shared_mutex in src/image.cpp (planning
        // exclusive, execution shared) -- no FFTW threads library is
        // needed.

        pyExportModel(m);
        pyExportImage(m);
        pyExportFpfs(m);
        math::pyExportMath(m);
        noise::pyExportNoise(m);
        mask::pyExportMask(m);
        ngmix::pyExportNgmix(m);
        table::pyExportTable(m);
        detector::pyExportDetector(m);
        geometry::pyExportGeometry(m);
        task::pyExportTask(m);
        psfmodel::pyExportPsfModel(m);
    }
}
