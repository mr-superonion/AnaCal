#include "anacal.h"


namespace anacal {

FpfsImage::FpfsImage(
    int nx,
    int ny,
    double scale,
    double sigma_arcsec,
    double klim,
    const py::array_t<double>& psf_array,
    bool use_estimate
): cimg(nx, ny, scale, use_estimate), psf_array(psf_array) {
    if ((sigma_arcsec <= 0) || (sigma_arcsec > 5.0)) {
        throw std::runtime_error(
            "FPFS Error: invalid input sigma_arcsec"
        );
    }
    this->nx = nx;
    this->ny = ny;
    this->scale = scale;
    this->sigma_arcsec = sigma_arcsec;
    this->klim = klim;
    this->sigma_f = 1.0 / sigma_arcsec;
    this->fft_ratio = 1.0 / scale / scale;
    return;
}


py::array_t<double>
FpfsImage::smooth_image(
    const py::array_t<double>& img_array,
    bool do_rotate,
    int x,
    int y
) {
    const Gaussian gauss_model(sigma_f);
    std::optional<py::array_t<double>> noise_conv;
    // Prepare PSF
    cimg.set_r(psf_array, -1, -1, true);
    cimg.fft();
    if (do_rotate) {
        cimg.rotate90_f();
    }
    const py::array_t<std::complex<double>> parr = cimg.draw_f();
    // signal
    cimg.set_r(img_array, x, y, false);
    cimg.fft();
    // Deconvolve the PSF
    cimg.deconvolve(parr, klim);
    // Convolve Gaussian
    cimg.filter(gauss_model);
    // Back to Real space
    cimg.ifft();
    py::array_t<double> img_conv = cimg.draw_r();
    return img_conv;
}


py::array_t<double>
FpfsImage::smooth_image(
    const py::array_t<double>& gal_array,
    const std::optional<py::array_t<double>>& noise_array,
    int x,
    int y
) {
    //TODO: use different PSF here
    py::array_t<double> gal_conv = this->smooth_image(
        gal_array,
        false,
        x,
        y
    );
    if (noise_array.has_value()) {
        py::array_t<double> noise_conv = this->smooth_image(
            *noise_array,
            true,
            x,
            y
        );
        auto g_r = gal_conv.mutable_unchecked<2>();
        auto n_r = noise_conv.mutable_unchecked<2>();
        for (ssize_t j = 0; j < this->ny; ++j) {
            for (ssize_t i = 0; i < this->nx; ++i) {
                g_r(j, i) = g_r(j, i) + n_r(j, i);
            }
        }
    }
    return gal_conv;
}


py::array_t<FpfsPeaks>
FpfsImage::find_peak(
    const py::array_t<double>& gal_conv,
    double fthres,
    double pthres,
    double std_m00,
    double std_v,
    int bound
) {
    auto r = gal_conv.unchecked<2>();
    ssize_t ny = r.shape(0);
    ssize_t nx = r.shape(1);

    double fcut = fthres * std_m00;
    double pcut = fpfs_pnr * std_v;
    double sigma_v = fpfs_cut_sigma_ratio * std_v;

    double wdet_cut = pthres - fpfs_det_sigma2 - 0.02;
    if (std::fabs(wdet_cut) < 1e-10) {
        wdet_cut = 0.0;
    }
    if (wdet_cut < 0.0) {
        throw std::runtime_error(
            "FPFS Error: The second pooling threshold pthres is too small."
        );
    }

    std::vector<std::tuple<int, int, bool>> peaks;
    for (ssize_t j = bound + 1; j < ny - bound; ++j) {
        for (ssize_t i = bound + 1; i < nx - bound; ++i) {
            double c = r(j, i);
            double d1 = c - r(j, i+1);
            double d2 = c - r(j+1, i);
            double d3 = c - r(j, i-1);
            double d4 = c - r(j-1, i);
            double s1 = math::ssfunc2(d1, sigma_v - pcut, sigma_v);
            double s2 = math::ssfunc2(d2, sigma_v - pcut, sigma_v);
            double s3 = math::ssfunc2(d3, sigma_v - pcut, sigma_v);
            double s4 = math::ssfunc2(d4, sigma_v - pcut, sigma_v);
            double wdet = s1 * s2 * s3 * s4;
            bool sel = (
                (c > fcut) &&
                (wdet > wdet_cut)
            );
            if (sel) {
                bool is_peak = (
                    (c > r(j-1, i)) &&
                    (c > r(j+1, i)) &&
                    (c > r(j, i-1)) &&
                    (c > r(j, i+1))
                );
                peaks.push_back({j, i, is_peak});
            }
        }
    }

    int nrow = peaks.size();
    py::array_t<FpfsPeaks> src(nrow);
    auto src_r = src.mutable_unchecked<1>();

    for (ssize_t j = 0; j < nrow; ++j) {
        const auto& elem = peaks[j];
        src_r(j).y = std::get<0>(elem);
        src_r(j).x = std::get<1>(elem);
        src_r(j).is_peak = int(std::get<2>(elem));
        src_r(j).mask_value = 0;
    }

    return src;
}


py::array_t<FpfsPeaks>
FpfsImage::detect_source(
    py::array_t<double>& gal_array,
    double fthres,
    double pthres,
    double std_m00,
    double std_v,
    int bound,
    const std::optional<py::array_t<double>>& noise_array,
    const std::optional<py::array_t<int16_t>>& mask_array
) {

    int nn = 256;
    auto r = gal_array.unchecked<2>();
    int npix_y = r.shape(0);
    int npix_x = r.shape(1);
    int my = npix_y / (nn - 64) + 1;
    int mx = npix_x / (nn - 64) + 1;
    py::array_t<double> gal_conv = this->smooth_image(
        gal_array,
        noise_array,
        -1,
        -1
    );

    py::array_t<FpfsPeaks> detection = this->find_peak(
        gal_conv,
        fthres,
        pthres,
        std_m00,
        std_v,
        bound
    );

    if (mask_array.has_value()) {
        add_pixel_mask_column(
            detection,
            *mask_array,
            sigma_arcsec,
            scale
        );
    }
    return detection;
}

py::array_t<double>
FpfsImage::measure_source(
    const py::array_t<double>& gal_array,
    const py::array_t<std::complex<double>>& filter_image,
    const std::optional<py::array_t<double>>& psf_array,
    const std::optional<py::array_t<FpfsPeaks>>& det,
    bool do_rotate
) {
    ssize_t ndim = filter_image.ndim();
    if ( ndim != 3) {
        throw std::runtime_error(
            "FPFS Error: Input filter image must be 3-dimensional."
        );
    }


    const py::array_t<double>&
        psf_use = psf_array.has_value() ? *psf_array : this->psf_array;
    cimg.set_r(psf_use, -1, -1, false);
    cimg.fft();
    if (do_rotate){
        cimg.rotate90_f();
    }
    const py::array_t<std::complex<double>> parr = cimg.draw_f();
    const py::array_t<std::complex<double>> fimg = deconvolve_filter(
        filter_image,
        parr,
        scale,
        klim
    );

    ssize_t ncol = filter_image.shape()[ndim - 1];
    py::array_t<FpfsPeaks> det_default(1);
    auto r = det_default.mutable_unchecked<1>();
    r(0).y = ny / 2; r(0).x = nx / 2;
    r(0).is_peak = 1; r(0).mask_value = 0;
    const py::array_t<FpfsPeaks>& det_use = det.has_value() ? *det : det_default;
    auto det_r = det_use.unchecked<1>();

    ssize_t nrow = det_use.shape()[0];
    py::array_t<double> src({nrow, ncol});
    auto src_r = src.mutable_unchecked<2>();
    for (ssize_t j = 0; j < nrow; ++j) {
        int y = det_r(j).y; int x = det_r(j).x;
        cimg.set_r(gal_array, x, y, false);
        cimg.fft();
        py::array_t<double> row = cimg.measure(fimg);
        auto row_r = row.unchecked<1>();
        for (ssize_t i = 0; i < ncol; ++i) {
            src_r(j, i) = row_r(i) * fft_ratio;
        }
    }
    return src;
}

py::array_t<double>
FpfsImage::measure_source(
    const py::array_t<double>& gal_array,
    const py::array_t<std::complex<double>>& filter_image,
    const BasePsf& psf_obj,
    const std::optional<py::array_t<FpfsPeaks>>& det,
    bool do_rotate
) {
    ssize_t ndim = filter_image.ndim();
    if ( ndim != 3) {
        throw std::runtime_error(
            "FPFS Error: Input filter image must be 3-dimensional."
        );
    }
    ssize_t ncol = filter_image.shape()[ndim - 1];

    py::array_t<FpfsPeaks> det_default(1);
    auto r = det_default.mutable_unchecked<1>();
    r(0).y = ny / 2; r(0).x = nx / 2;
    r(0).is_peak = 1; r(0).mask_value = 0;
    const py::array_t<FpfsPeaks>& det_use = det.has_value() ? *det : det_default;
    auto det_r = det_use.unchecked<1>();

    ssize_t nrow = det_use.shape()[0];
    py::array_t<double> src({nrow, ncol});
    auto src_r = src.mutable_unchecked<2>();
    for (ssize_t j = 0; j < nrow; ++j) {
        int y = det_r(j).y; int x = det_r(j).x;
        {
            py::array_t<double> psf_use = psf_obj.draw(x, y);
            cimg.set_r(psf_use, -1, -1, false);
        }
        cimg.fft();
        if (do_rotate){
            cimg.rotate90_f();
        }
        {
            const py::array_t<std::complex<double>> parr = cimg.draw_f();
            cimg.set_r(gal_array, x, y, false);
            cimg.fft();
            cimg.deconvolve(parr, klim);
        }
        py::array_t<double> row = cimg.measure(filter_image);
        auto row_r = row.unchecked<1>();
        for (ssize_t i = 0; i < ncol; ++i) {
            src_r(j, i) = row_r(i) * fft_ratio;
        }
    }
    return src;
}


FpfsImage::~FpfsImage() {
}


void
pyExportFpfs(py::module& m) {
    py::module_ fpfs = m.def_submodule("fpfs", "submodule for FPFS shear estimation");
    fpfs.attr("fpfs_cut_sigma_ratio") = fpfs_cut_sigma_ratio;
    fpfs.attr("fpfs_det_sigma2") = fpfs_det_sigma2;
    fpfs.attr("fpfs_pnr") = fpfs_pnr;
    py::class_<FpfsImage>(fpfs, "FpfsImage")
        .def(py::init<int, int, double, double, double,
            const py::array_t<double>&, bool>(),
            "Initialize the FpfsImage object using an ndarray",
            py::arg("nx"), py::arg("ny"),
            py::arg("scale"),
            py::arg("sigma_arcsec"),
            py::arg("klim"),
            py::arg("psf_array"),
            py::arg("use_estimate")=true
        )
        .def("smooth_image",
            py::overload_cast<
                const py::array_t<double>&,
                bool,
                int,
                int
            >(&FpfsImage::smooth_image),
            "Smooths the image after PSF deconvolution",
            py::arg("gal_array"),
            py::arg("do_rotate")=false,
            py::arg("x")=-1,
            py::arg("y")=-1
        )
        .def("smooth_image",
            py::overload_cast<
                const py::array_t<double>&,
                const std::optional<py::array_t<double>>&,
                int,
                int
            >(&FpfsImage::smooth_image),
            "Smooths the image after PSF deconvolution",
            py::arg("gal_array"),
            py::arg("noise_array")=py::none(),
            py::arg("x")=-1,
            py::arg("y")=-1
        )
        .def("find_peak", &FpfsImage::find_peak,
            "Detects peaks from smoothed images",
            py::arg("gal_conv"),
            py::arg("fthres"),
            py::arg("pthres"),
            py::arg("std_m00"),
            py::arg("std_v"),
            py::arg("bound")
        )
        .def("detect_source",
            &FpfsImage::detect_source,
            "Detect galaxy candidates from image",
            py::arg("gal_array"),
            py::arg("fthres"),
            py::arg("pthres"),
            py::arg("std_m00"),
            py::arg("std_v"),
            py::arg("bound"),
            py::arg("noise_array")=py::none(),
            py::arg("mask_array")=py::none()
        )
        .def("measure_source",
            py::overload_cast<
                const py::array_t<double>&,
                const py::array_t<std::complex<double>>&,
                const std::optional<py::array_t<double>>&,
                const std::optional<py::array_t<FpfsPeaks>>&,
                bool
            >(&FpfsImage::measure_source),
            "measure source properties using filter at the position of det",
            py::arg("gal_array"),
            py::arg("filter_image"),
            py::arg("psf_array")=py::none(),
            py::arg("det")=py::none(),
            py::arg("do_rotate")=false
        )
        .def("measure_source",
            py::overload_cast<
                const py::array_t<double>&,
                const py::array_t<std::complex<double>>&,
                const BasePsf&,
                const std::optional<py::array_t<FpfsPeaks>>&,
                bool
            >(&FpfsImage::measure_source),
            "measure source properties using filter at the position of det",
            py::arg("gal_array"),
            py::arg("filter_image"),
            py::arg("psf_obj"),
            py::arg("det")=py::none(),
            py::arg("do_rotate")=false
        );
}
}
