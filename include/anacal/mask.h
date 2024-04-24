#ifndef ANACAL_MASK_H
#define ANACAL_MASK_H

namespace anacal {
    void
    add_pixel_mask_value(
        py::array_t<int>& det,
        const py::array_t<int>& mask_array,
        double sigma,
        double scale,
        int bound
    );

    py::array_t<float>
    smooth_mask_image(
        const py::array_t<int>& mask_array,
        double sigma,
        double scale,
        int bound
    );

    void pyExportMask(py::module& m);
}

#endif // MASK_H
