#ifndef ANACAL_MASK_H
#define ANACAL_MASK_H

namespace anacal {
    void
    add_pixel_mask_column(
        py::array_t<int>& det,
        const py::array_t<int16_t>& mask_array,
        double sigma,
        double scale,
        int bound
    );

    py::array_t<float>
    smooth_mask_image(
        const py::array_t<int16_t>& mask_array,
        double sigma,
        double scale,
        int bound
    );

    void
    mask_bright_stars(
        py::array_t<int16_t>& mask_array,
        const py::array_t<float>& x_array,
        const py::array_t<float>& y_array,
        const py::array_t<float>& r_array
    );

    void pyExportMask(py::module& m);
}

#endif // MASK_H
