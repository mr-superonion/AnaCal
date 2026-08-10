#include "anacal.h"


namespace anacal {
    PYBIND11_MODULE(_anacal, m)
    {
        // Thread safety note: the measurement runners release the GIL
        // around their compute, so several Python threads run the C++
        // core at once.  The FFT engine (pocketfft, see image.h) keeps
        // no global state and takes no lock, so that region is fully
        // lock-free -- no external FFT library is needed at all.

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
