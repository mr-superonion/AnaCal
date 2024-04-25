#include "anacal.h"


namespace anacal {
    void
    add_pixel_mask_value(
        py::array_t<int>& det,
        const py::array_t<int16_t>& mask_array,
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

        auto det_r = det.mutable_unchecked<2>();
        ssize_t nrow = det_r.shape(0);
        for (ssize_t j = 0; j < nrow; ++j) {
            int y = det_r(j, 0); int x = det_r(j, 1);
            if (y>=0 && y< ny && x>=0 && x<nx) {
                det_r(j, 3) = int(conv_r(y, x) * 1000);
                /* std::cout<<conv_r(y, x) * 1000<<std::endl; */
            }
        }
        return;
    }


    py::array_t<float>
    smooth_mask_image(
        const py::array_t<int16_t>& mask_array,
        double sigma,
        double scale,
        int bound
    ) {
        int ngrid = int(sigma / scale) * 6 + 1;
        int ngrid2 = int((ngrid - 1) / 2);
        if (bound <= ngrid2) {
            throw std::runtime_error("Parameter Error: bound too small");
        }

        py::array_t<float> kernel({ngrid, ngrid});
        auto kernel_r = kernel.mutable_unchecked<2>();
        // Compute the Gaussian kernel
        float A = float(scale * scale / (2.0 * M_PI * sigma * sigma));
        float sigma2 = -1.0 / (2 * float(sigma * sigma));
        for (int y = 0; y < ngrid; ++y) {
            for (int x = 0; x < ngrid; ++x) {
                float dx = (x - ngrid2) * scale;
                float dy = (y - ngrid2) * scale;
                float r2 = dx * dx + dy * dy;
                kernel_r(y, x) = A * std::exp(r2 * sigma2);
            }
        }

        auto mask_r = mask_array.unchecked<2>();
        int ny = mask_r.shape(0);
        int nx = mask_r.shape(1);
        py::array_t<float> mask_conv({ny, nx});
        auto conv_r = mask_conv.mutable_unchecked<2>();
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                conv_r(j, i) = 0.0;
            }
        }

        for (int j = bound; j < ny - bound; ++j) {
            for (int i = bound; i < nx - bound; ++i) {
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
