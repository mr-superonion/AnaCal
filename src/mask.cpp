#include "anacal.h"


namespace anacal {
    void
    add_bright_star_mask(
        py::array_t<int16_t>& mask_array,
        const py::array_t<BrightStar>& star_array
    ) {
        auto star_r = star_array.unchecked<1>();
        int nn = star_array.shape(0);
        int ndim = mask_array.ndim();
        if (ndim != 2) {
            throw std::runtime_error(
                "Mask Error: The input mask array has an invalid shape."
            );
        }
        auto m_r = mask_array.mutable_unchecked<2>();
        int ny = m_r.shape(0);
        int nx = m_r.shape(1);
        for (int k = 0; k < nn; ++k) {
            int x = static_cast<int>(star_r(k).x + 0.5);
            int y = static_cast<int>(star_r(k).y + 0.5);
            int r = static_cast<int>(star_r(k).r + 0.5);
            int r2 = r * r;
            for (int j = y-r; j <= y+r; ++j) {
                if ((j < 0) || (j >= ny)) {
                    continue;
                }
                int dy2 = (j - y) * (j - y);
                for (int i = x-r; i <= x+r; ++i) {
                    if ((i < 0) || (i >= nx)) {
                        continue;
                    }
                    int dx2 = (i - x) * (i - x);
                    int d2 = dx2 + dy2;
                    if (d2 < r2) {
                        m_r(j, i) = m_r(j, i) | 1;
                    }
                }
            }
        }
        return;
    }

    void
    extend_mask_image(
        py::array_t<int16_t>& mask_array
    ) {
        // Get information of input array
        int ndim = mask_array.ndim();
        if (ndim != 2) {
            throw std::runtime_error(
                "Mask Error: The input mask array has an invalid shape."
            );
        }
        auto m_r = mask_array.mutable_unchecked<2>();
        int ny = m_r.shape(0);
        int nx = m_r.shape(1);

        // Initialize the convolved mask
        py::array_t<int16_t> mask_conv({ny, nx});
        auto conv_r = mask_conv.mutable_unchecked<2>();
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                conv_r(j, i) = 0.0;
            }
        }

        // Convoved image
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (m_r(y, x) > 0){
                    for (int j = y-2; j <= y+2; ++j) {
                        if ((j < 0) || (j >= ny)) {
                            continue;
                        }
                        for (int i = x-2; i <= x+2; ++i) {
                            if ((i < 0) || (i >= nx)) {
                                continue;
                            }
                            conv_r(j, i) = 1;
                        }
                    }
                }
            }
        }
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                m_r(j, i) = conv_r(j, i);
            }
        }
        return;
    }


    void
    mask_galaxy_image(
        py::array_t<double>& gal_array,
        py::array_t<int16_t>& mask_array,
        bool do_extend_mask,
        const std::optional<py::array_t<BrightStar>>& star_array
    ) {
        if (do_extend_mask) {
            extend_mask_image(mask_array);
        }
        if (star_array.has_value()) {
            add_bright_star_mask(
                mask_array,
                *star_array
            );
        }

        auto img_r = gal_array.mutable_unchecked<2>();
        int ny = gal_array.shape(0);
        int nx = gal_array.shape(1);
        auto mask_r = mask_array.unchecked<2>();

        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                if (mask_r(j, i) > 0) {
                    img_r(j, i) = 0.0;
                }
            }
        }
    }

    py::array_t<float>
    smooth_mask_image(
        const py::array_t<int16_t>& mask_array,
        double sigma,
        double scale
    ) {
        int ngrid = int(sigma / scale) * 6 + 1;
        int ngrid2 = int((ngrid - 1) / 2);

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

        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                if (mask_r(y, x) > 0) {
                    for (int j = -ngrid2; j <= ngrid2; ++j) {
                        if ((y + j < 0) || (y + j >= ny)) {
                            continue;
                        }
                        for (int i = -ngrid2; i <= ngrid2; ++i) {
                            if ((x + i < 0) || (x + i >= nx)) {
                                continue;
                            }
                            conv_r(y + j, x + i) += (
                                mask_r(y, x) * kernel_r(j + ngrid2, i + ngrid2)
                            );
                        }
                    }
                }
            }
        }
        return mask_conv;
    }

    py::array_t<FpfsPeaks>
    add_pixel_mask_column(
        const py::array_t<FpfsPeaks>& det,
        const py::array_t<int16_t>& mask_array,
        double sigma,
        double scale
    ) {
        py::array_t<float> mask_conv = smooth_mask_image(
            mask_array, sigma, scale
        );

        auto conv_r = mask_conv.unchecked<2>();
        int ny = conv_r.shape(0);
        int nx = conv_r.shape(1);

        auto det_r = det.unchecked<1>();
        ssize_t nrow = det_r.shape(0);

        py::array_t<FpfsPeaks> out(nrow);
        auto out_r = out.template mutable_unchecked<1>();

        for (ssize_t j = 0; j < nrow; ++j) {
            out_r(j).mask_value = det_r(j).mask_value;
            out_r(j).is_peak = det_r(j).is_peak;
            out_r(j).y = det_r(j).y;
            out_r(j).x = det_r(j).x;
            int y = static_cast<int>(std::round(det_r(j).y));
            int x = static_cast<int>(std::round(det_r(j).x));
            if (y>=0 && y< ny && x>=0 && x<nx) {
                out_r(j).mask_value = int(conv_r(y, x) * 1000);
                /* std::cout<<det_r(j).mask_value<<std::endl; */
            }
        }
        return out;
    }

    void
    pyExportMask(py::module& m) {
        PYBIND11_NUMPY_DTYPE(BrightStar, x, y, r);
        PYBIND11_NUMPY_DTYPE(
            FpfsPeaks,
            y, x,
            is_peak,
            mask_value
        );
        py::module_ mask = m.def_submodule("mask", "submodule for mask");
        mask.def(
            "add_bright_star_mask", &add_bright_star_mask,
            "Update mask image according to bright star catalog",
            py::arg("mask_array"),
            py::arg("star_array")
        );
        mask.def(
            "extend_mask_image", &extend_mask_image,
            "Update mask image with a 2 pixel extension",
            py::arg("mask_array")
        );
        mask.def(
            "mask_galaxy_image", &mask_galaxy_image,
            "Apply mask on galaxy image",
            py::arg("gal_array"),
            py::arg("mask_array"),
            py::arg("do_extend_mask")=true,
            py::arg("star_array")=py::none()
        );
        mask.def(
            "smooth_mask_image", &smooth_mask_image,
            "Smooths the mask image",
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale")
        );
        mask.def(
            "add_pixel_mask_column", &add_pixel_mask_column,
            "Update the detection catalog with the pixel mask value",
            py::arg("det"),
            py::arg("mask_array"),
            py::arg("sigma"),
            py::arg("scale")
        );
    }
}
