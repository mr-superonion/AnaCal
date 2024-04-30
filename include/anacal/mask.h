#ifndef ANACAL_MASK_H
#define ANACAL_MASK_H

namespace anacal {
    struct BrightStar {
        float x;
        float y;
        float r;
    };


    void
    add_bright_star_mask(
        py::array_t<int16_t>& mask_array,
        const py::array_t<BrightStar>& star_array
    );

    void
    extend_mask_image(
        py::array_t<int16_t>& mask_array
    );


    void
    mask_galaxy_image(
        py::array_t<double>& gal_array,
        py::array_t<int16_t>& mask_array,
        bool do_extend_mask=true,
        const std::optional<py::array_t<BrightStar>>& star_array=std::nullopt
    );

    py::array_t<float>
    smooth_mask_image(
        const py::array_t<int16_t>& mask_array,
        double sigma,
        double scale
    );

    void
    add_pixel_mask_column(
        py::array_t<int>& det,
        const py::array_t<int16_t>& mask_array,
        double sigma,
        double scale
    );

    void pyExportMask(py::module& m);
}

#endif // MASK_H
