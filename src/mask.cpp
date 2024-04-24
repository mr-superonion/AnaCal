#include "anacal.h"


namespace anacal {
    void
    add_pixel_mask_value(
        std::vector<std::tuple<int, int, bool, int>>& det,
        const py::array_t<int>& mask_array,
        double sigma,
        double scale,
        int bound
    ) {
        py::array_t<float> mask_conv = smooth_mask_image(
            mask_array, sigma, scale, bound
        );

        auto conv_r = mask_conv.unchecked<2>();
        int ny = conv_r.shape(0);
        int nx = conv_r.shape(1);
        ssize_t nrow = det.size();
        for (ssize_t j = 0; j < nrow; ++j) {
            auto& elem = det[j];
            int y = std::get<0>(elem);
            int x = std::get<1>(elem);
            if (y>=0 && y< ny && x>=0 && x<nx) {
                std::get<3>(elem) = int(conv_r(y, x) * 1000);
            }
        }
        return;
    }


    py::array_t<float>
    smooth_mask_image(
        const py::array_t<int>& mask_array,
        double sigma,
        double scale,
        int bound
    ) {
        int ngrid = int(sigma / scale) * 6 + 1;
        float A = 1.0f / (2.0f * M_PI * float(sigma * sigma));

        py::array_t<float> kernel({ngrid, ngrid});
        auto kernel_r = kernel.mutable_unchecked<2>();
        int ngrid2 = int((ngrid - 1) / 2);
        if (bound <= ngrid2) {
            throw std::runtime_error("Parameter Error: bound too small");
        }

        // Compute the Gaussian kernel
        for (int y = 0; y < ngrid; ++y) {
            for (int x = 0; x < ngrid; ++x) {
                float dx = x - ngrid2;
                float dy = y - ngrid2;
                float r2 = dx * dx + dy * dy;
                kernel_r(y, x) = A * std::exp(-r2 / (2 * sigma * sigma));
            }
        }

        auto mask_r = mask_array.unchecked<2>();
        py::array_t<float> mask_conv(
            py::array::ShapeContainer(
                {mask_array.shape(0), mask_array.shape(1)}
            )
        );
        auto conv_r = mask_conv.mutable_unchecked<2>();

        for (int j = bound; j < ngrid - bound; ++j) {
            for (int i = bound; i < ngrid - bound; ++i) {
                if (mask_r(j, i) > 0) {
                    for (int jj = -ngrid2; jj <= ngrid2; ++jj) {
                        for (int ii = -ngrid2; ii <= ngrid2; ++ii) {
                            conv_r(j+jj, i+ii) += (
                                mask_r(j, i) * kernel_r(jj+ngrid2, ii+ngrid2)
                            );
                        }
                    }
                }
            }
        }
        return mask_conv;
    }


    void
    pyExportMask(py::module& m) {
        py::module_ mask = m.def_submodule("mask", "submodule for mask");
        mask.def(
            "add_pixel_mask_value", &add_pixel_mask_value,
            "Measures the pixel mask value",
            py::arg("det"),
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale"),
            py::arg("bound")
        );
        mask.def(
            "smooth_mask_image", &smooth_mask_image,
            "Smooths the mask plane",
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale"),
            py::arg("bound")
        );
    }
}
