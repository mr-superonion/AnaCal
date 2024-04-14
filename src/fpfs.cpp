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
        throw std::runtime_error("Error: wrong sigma_arcsec");
    }
    this->nx = nx;
    this->ny = ny;
    this->scale = scale;
    this->sigma_arcsec = sigma_arcsec;
    this->klim = klim;
    sigma_f = 1.0 / sigma_arcsec;
    fft_ratio = 1.0 / scale / scale;
    return;
}


py::array_t<double>
FpfsImage::smooth_image(
    const py::array_t<double>& gal_array,
    const std::optional<py::array_t<double>>& noise_array
) {
    const Gaussian gauss_model(sigma_f);

    // Psf
    cimg.set_r(psf_array, true);
    cimg.fft();
    {
        const py::array_t<std::complex<double>> parr1 = cimg.draw_f();
        if (noise_array.has_value()) {
            cimg.rotate90_f();
            {
                const py::array_t<std::complex<double>> parr2 = cimg.draw_f();

                cimg.set_r(*noise_array, false);
                cimg.fft();
                cimg.deconvolve(parr2, klim);
            }
            cimg.filter(gauss_model);

            const py::array_t<std::complex<double>> narr = cimg.draw_f();
            // Galaxy
            cimg.set_r(gal_array, false);
            cimg.fft();

            cimg.deconvolve(parr1, klim);
            cimg.filter(gauss_model);
            cimg.add_image_f(narr);
        } else {
            cimg.set_r(gal_array, false);
            cimg.fft();

            cimg.deconvolve(parr1, klim);
            cimg.filter(gauss_model);
        }
    }
    // Galaxy
    cimg.ifft();
    py::array_t<double> gal_conv = cimg.draw_r();
    return gal_conv;
}


std::vector<std::tuple<int, int, bool>>
FpfsImage::find_peak(
    const py::array_t<double>& gal_conv,
    double fthres,
    double pthres,
    double pratio,
    double std_m00,
    double std_v,
    int bound,
    double wdet_cut
) {
    auto r = gal_conv.unchecked<2>();
    ssize_t ny = r.shape(0);
    ssize_t nx = r.shape(1);

    double fcut = fthres * std_m00;
    double pcut = pthres * std_v;
    double sigma_v = fpfs_cut_sigma_ratio * std_v;

    std::vector<std::tuple<int, int, bool>> peaks;
    for (ssize_t j = bound + 1; j < ny - bound; ++j) {
        for (ssize_t i = bound + 1; i < nx - bound; ++i) {
            double c = r(j, i);
            double thres = pcut + pratio * c;
            double d1 = c - r(j, i+1);
            double d2 = c - r(j+1, i);
            double d3 = c - r(j, i-1);
            double d4 = c - r(j-1, i);
            double s1 = imath::ssfunc2(d1, sigma_v - thres, sigma_v);
            double s2 = imath::ssfunc2(d2, sigma_v - thres, sigma_v);
            double s3 = imath::ssfunc2(d3, sigma_v - thres, sigma_v);
            double s4 = imath::ssfunc2(d4, sigma_v - thres, sigma_v);
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
    return peaks;
}


std::vector<std::tuple<int, int, bool>>
FpfsImage::detect_source(
    const py::array_t<double>& gal_array,
    double fthres,
    double pthres,
    double pratio,
    double std_m00,
    double std_v,
    int bound,
    const std::optional<py::array_t<double>>& noise_array,
    double wdet_cut
) {

    py::array_t<double> gal_conv = smooth_image(
        gal_array,
        noise_array
    );
    auto catalog = find_peak(
        gal_conv,
        fthres,
        pthres,
        pratio,
        std_m00,
        std_v,
        bound,
        wdet_cut
    );
    return catalog;
}

py::array_t<double>
FpfsImage::measure_source(
    const py::array_t<double>& gal_array,
    const py::array_t<std::complex<double>>& filter_image,
    const std::optional<py::array_t<double>>& psf_array,
    const std::optional<std::vector<std::tuple<int, int, bool>>>& det,
    bool do_rotate
) {
    ssize_t ndim = filter_image.ndim();
    if ( ndim != 3) {
        throw std::runtime_error("Error: Input must be 3-dimensional.");
    }


    const py::array_t<double>&
        psf_use = psf_array.has_value() ? *psf_array : this->psf_array;
    cimg.set_r(psf_use, false);
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
    const std::vector<std::tuple<int, int, bool>>
        det_default = {{ny/2, nx/2, false}};
    const std::vector<std::tuple<int, int, bool>>&
        det_use = det.has_value() ? *det : det_default;
    ssize_t nrow = det_use.size();

    py::array_t<double> src({nrow, ncol});
    auto src_r = src.mutable_unchecked<2>();
    for (ssize_t j = 0; j < nrow; ++j) {
        const auto& elem = det_use[j];
        int y = std::get<0>(elem);
        int x = std::get<1>(elem);
        cimg.set_r(gal_array, x, y);
        cimg.fft();
        /* cimg.deconvolve(parr, klim); */
        py::array_t<double> row = cimg.measure(fimg);
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
    py::class_<FpfsImage>(fpfs, "FpfsImage")
        .def(py::init<int, int, double, double, double,
            const py::array_t<double>&, bool>(),
            "Initialize the FpfsImage object using an ndarray",
            py::arg("nx"), py::arg("ny"),
            py::arg("scale"), py::arg("sigma_arcsec"),
            py::arg("klim"),
            py::arg("psf_array"),
            py::arg("use_estimate")=true
        )
        .def("smooth_image", &FpfsImage::smooth_image,
            "Smooths the image after PSF deconvolution",
            py::arg("gal_array"),
            py::arg("noise_array")=py::none()
        )
        .def("find_peak", &FpfsImage::find_peak,
            "Detects peaks from smoothed images",
            py::arg("gal_conv"),
            py::arg("fthres"),
            py::arg("pthres"),
            py::arg("pratio"),
            py::arg("std_m00"),
            py::arg("std_v"),
            py::arg("bound"),
            py::arg("wdet_cut")=0.0
        )
        .def("detect_source", &FpfsImage::detect_source,
            "Detect galaxy candidates from image",
            py::arg("gal_array"),
            py::arg("fthres"),
            py::arg("pthres"),
            py::arg("pratio"),
            py::arg("std_m00"),
            py::arg("std_v"),
            py::arg("bound"),
            py::arg("noise_array")=py::none(),
            py::arg("wdet_cut")=0.0
        )
        .def("measure_source", &FpfsImage::measure_source,
            "measure source properties using filter at the position of det",
            py::arg("gal_array"),
            py::arg("filter_image"),
            py::arg("psf_array")=py::none(),
            py::arg("det")=py::none(),
            py::arg("do_rotate")=false
        );
}
}
