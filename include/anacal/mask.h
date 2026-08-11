#ifndef ANACAL_MASK_H
#define ANACAL_MASK_H

#include "table.h"

namespace anacal {
namespace mask {
    // Every mask argument below is EDITED IN PLACE, so its dtype has to
    // be checked by hand before pybind is allowed near it.  Declaring
    // the argument as ``py::array_t<int16_t>`` would let pybind accept
    // any dtype and hand us a converted *copy*: the flags would be set
    // in the copy and the caller's mask would come back untouched, with
    // no error raised.  Taking an untyped ``py::array`` skips that
    // conversion; this helper turns it back into a typed handle that
    // still refers to the CALLER's buffer.
    inline py::array_t<int16_t>
    borrow_mask(py::array& mask_array_in, const char* who) {
        if (!mask_array_in.dtype().is(py::dtype::of<int16_t>())) {
            throw std::invalid_argument(
                std::string(who) + " Error: mask_array is modified in "
                "place and must already be int16; got dtype '" +
                py::str(mask_array_in.dtype()).cast<std::string>() + "'"
            );
        }
        if (mask_array_in.ndim() != 2) {
            throw std::runtime_error(
                "Mask Error: The input mask array has an invalid shape."
            );
        }
        return py::reinterpret_borrow<py::array_t<int16_t>>(mask_array_in);
    }

    // Flags the star footprints into ``mask_array`` (already validated).
    void
    inline add_bright_star_mask_impl(
        py::array_t<int16_t>& mask_array,
        const py::array_t<BrightStar>& star_array
    ) {
        auto star_r = star_array.unchecked<1>();
        int nn = star_array.shape(0);
        auto m_r = mask_array.mutable_unchecked<2>();
        int ny = m_r.shape(0);
        int nx = m_r.shape(1);
        for (int k = 0; k < nn; ++k) {
            int x = static_cast<int>(star_r(k).x + 0.5);
            int y = static_cast<int>(star_r(k).y + 0.5);
            float radius = star_r(k).r;
            if (!std::isfinite(radius) || radius < 0.0f) {
                continue;
            }
            int r = static_cast<int>(std::round(radius));
            if (r == 0) {
                r = 1;
            }
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
    };

    // Public entry point: validates, then flags the star footprints.
    void
    inline add_bright_star_mask(
        py::array& mask_array_in,
        const py::array_t<BrightStar>& star_array
    ) {
        auto mask_array = borrow_mask(mask_array_in, "add_bright_star_mask");
        add_bright_star_mask_impl(mask_array, star_array);
        return;
    };

    // BOTH array arguments are edited in place: the galaxy pixels under
    // a set mask flag are zeroed, and -- when a star catalog is given --
    // the star footprints are flagged into mask_array.
    void
    inline mask_galaxy_image(
        py::array& gal_array_in,
        py::array& mask_array_in,
        const std::optional<py::array_t<BrightStar>>& star_array
    ) {
        // Same in-place dtype trap as the mask (see borrow_mask): pybind
        // would hand us a converted *copy* of a non-float32 array, zero
        // the pixels there, and leave the caller's image untouched with
        // no error raised.  Taking an untyped ``py::array`` skips that
        // conversion entirely.
        if (!gal_array_in.dtype().is(py::dtype::of<pixel_t>())) {
            throw std::invalid_argument(
                "mask_galaxy_image Error: gal_array is modified in place and "
                "must already be float32; got dtype '" +
                py::str(gal_array_in.dtype()).cast<std::string>() + "'"
            );
        }
        auto gal_array = py::reinterpret_borrow<py::array_t<pixel_t>>(
            gal_array_in
        );
        auto mask_array = borrow_mask(mask_array_in, "mask_galaxy_image");

        if (star_array.has_value()) {
            add_bright_star_mask_impl(
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
                    img_r(j, i) = static_cast<pixel_t>(0.0);
                }
            }
        }
    }

    py::array_t<float>
    inline convolve_mask_gauss(
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
                            // Convolution of the 0/1 mask with the kernel.
                            // The pixel value is not used as a factor: if the
                            // input were ever a raw bitmask (SAT=2, CR=8 ...)
                            // each pixel would be weighted by its bit value,
                            // which has no meaning.  Any pixel that passes the
                            // mask_r > 0 gate counts as exactly one.
                            conv_r(y + j, x + i) += kernel_r(
                                j + ngrid2, i + ngrid2
                            );
                        }
                    }
                }
            }
        }
        return mask_conv;
    }

    inline py::array_t<int> convolve_mask(
        py::array_t<int> mask_array,
        py::array_t<int> kernel
    ) {
        auto img = mask_array.unchecked<2>();
        auto ker = kernel.unchecked<2>();

        const int height = img.shape(0);
        const int width  = img.shape(1);
        const int kh = ker.shape(0);
        const int kw = ker.shape(1);
        if (kh % 2 != 1 || kw % 2 != 1)
            throw std::invalid_argument("Kernel size must be odd");
        const int kh2 = kh / 2;
        const int kw2 = kw / 2;

        py::array_t<int> mask_conv({height, width});
        auto out = mask_conv.mutable_unchecked<2>();

        // Initialize mask_conv to zero
        for (int i = 0; i < height; ++i)
            for (int j = 0; j < width; ++j)
                out(i, j) = 0.0;

        // Loop over only nonzero pixels
        for (int y = kh2; y < height - kh2; ++y) {
            for (int x = kw2; x < width - kw2; ++x) {
                int v = img(y, x);
                if (v == 0.0) continue;  // Skip zeros
                for (int dy = -kh2; dy <= kh2; ++dy) {
                    const int ny = y + dy;
                    const int ky = dy + kh2;
                    for (int dx = -kw2; dx <= kw2; ++dx) {
                        const int nx = x + dx;
                        const int kx = dx + kw2;
                        out(ny, nx) += v * ker(ky, kx);
                    }
                }
            }
        }
        return mask_conv;
    };

    // Stamp the per-source mask value: the Gaussian-smoothed mask
    // (convolve_mask_gauss) sampled at the source centre, times 1000 --
    // the same values the old convolve-then-sample pair produced, bit for
    // bit (same kernel, and masked pixels are accumulated in the same
    // raster order).  The smoothed value is evaluated AT each source
    // position instead of over the whole image, so the caller's mask is
    // only read (never modified), no Python object is created -- safe
    // under a released GIL -- and the cost is O(sources * kernel^2), not
    // O(image) per call.
    void
    inline add_pixel_mask_column(
        std::vector<table::galNumber>& catalog,
        const py::array_t<int16_t>& mask_array,
        double sigma,
        double scale
    ) {
        // Kernel exactly as in convolve_mask_gauss.
        const int ngrid = int(sigma / scale) * 6 + 1;
        const int ngrid2 = int((ngrid - 1) / 2);
        std::vector<float> kernel(
            static_cast<std::size_t>(ngrid) * ngrid
        );
        const float A = float(scale * scale / (2.0 * M_PI * sigma * sigma));
        const float sigma2 = -1.0 / (2 * float(sigma * sigma));
        for (int y = 0; y < ngrid; ++y) {
            for (int x = 0; x < ngrid; ++x) {
                float dx = (x - ngrid2) * scale;
                float dy = (y - ngrid2) * scale;
                float r2 = dx * dx + dy * dy;
                kernel[y * ngrid + x] = A * std::exp(r2 * sigma2);
            }
        }

        auto mask_r = mask_array.unchecked<2>();
        const int ny = static_cast<int>(mask_r.shape(0));
        const int nx = static_cast<int>(mask_r.shape(1));

        for (table::galNumber & src : catalog) {
            const int y = static_cast<int>(
                std::round(src.model.x2.v / scale)
            );
            const int x = static_cast<int>(
                std::round(src.model.x1.v / scale)
            );
            if (y < 0 || y >= ny || x < 0 || x >= nx) {
                continue;
            }
            // conv(y, x) = sum over masked pixels (my, mx) within the
            // kernel reach of kernel(y - my, x - mx); accumulated in
            // float and in raster order of (my, mx) to reproduce the
            // full-image convolution exactly.
            float conv = 0.0f;
            const int j0 = std::max(y - ngrid2, 0);
            const int j1 = std::min(y + ngrid2, ny - 1);
            const int i0 = std::max(x - ngrid2, 0);
            const int i1 = std::min(x + ngrid2, nx - 1);
            for (int my = j0; my <= j1; ++my) {
                for (int mx = i0; mx <= i1; ++mx) {
                    if (mask_r(my, mx) > 0) {
                        conv += kernel[
                            (y - my + ngrid2) * ngrid + (x - mx + ngrid2)
                        ];
                    }
                }
            }
            src.mask_value = static_cast<int>(conv * 1000);
        }
        return;
    };

    void pyExportMask(py::module& m);

} // end mask
} // end anacal

#endif // MASK_H
