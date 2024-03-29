#include "anacal.h"


namespace anacal {

FpfsImage::FpfsImage(
    int nx,
    int ny,
    double scale,
    double sigma_arcsec,
    double klim,
    const py::array_t<double>& psf_array
): cimg(nx, ny, scale), psf_array(psf_array) {
    if ((sigma_arcsec <= 0) || (sigma_arcsec > 5.0)) {
        throw std::runtime_error("Error: wrong sigma_arcsec");
    }
    this->nx = nx;
    this->ny = ny;
    this->scale = scale;
    this->sigma_arcsec = sigma_arcsec;
    this->klim = klim;
    sigma_f = 1.0 / sigma_arcsec;
    return;
}


py::array_t<double>
FpfsImage::smooth_image(
    const py::array_t<double>& gal_array,
    const py::array_t<double>& noise_array
) {
    const Gaussian gauss_model(sigma_f);
    const ssize_t* shape_n = noise_array.shape();
    ssize_t ny_n = shape_n[0];
    ssize_t nx_n = shape_n[1];

    // Psf
    cimg.set_r(psf_array, true);
    cimg.fft();
    {
        const py::array_t<std::complex<double>> parr1 = cimg.draw_f();
        if ((ny_n == ny) && (nx_n == nx)) {
            cimg.rotate90_f();
            {
                const py::array_t<std::complex<double>> parr2 = cimg.draw_f();

                cimg.set_r(noise_array, false);
                cimg.fft();
                cimg.deconvolve(parr2, klim);
                cimg.filter(gauss_model);
            }
            py::array_t<std::complex<double>> narr = cimg.draw_f();

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
FpfsImage::find_peaks(
    const py::array_t<double>& gal_conv,
    double fthres,
    double pthres,
    double pratio,
    double std_m00,
    double std_v,
    int bound
) {
    auto r = gal_conv.unchecked<2>();
    ssize_t ny = r.shape(0);
    ssize_t nx = r.shape(1);

    double fcut = fthres * std_m00;
    double pcut = pthres * std_v;

    std::vector<std::tuple<int, int, bool>> peaks;
    for (ssize_t j = bound + 1; j < ny - bound; ++j) {
        for (ssize_t i = bound + 1; i < nx - bound; ++i) {
            double c = r(j, i);
            double thres = pcut + pratio * c;
            bool sel = (
                (c > fcut) &&
                (c > r(j-1, i) - thres) &&
                (c > r(j+1, i) - thres) &&
                (c > r(j, i-1) - thres) &&
                (c > r(j, i+1) - thres)
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


py::array_t<double>
FpfsImage::measure_sources(
    const py::array_t<double>& gal_array,
    const py::array_t<std::complex<double>>& filter_image,
    const std::vector<std::tuple<int, int, bool>>& det
) {
    ssize_t ndim = filter_image.ndim();
    if ( ndim != 3) {
        throw std::runtime_error("Error: Input must be 3-dimensional.");
    }
    ssize_t ncol = filter_image.shape()[0];
    ssize_t nrow = det.size();

    py::array_t<double> src({nrow, ncol});
    auto src_mut = src.mutable_unchecked<2>();

    cimg.set_r(psf_array, false);
    cimg.fft();
    const py::array_t<std::complex<double>> parr = cimg.draw_f();
    const double ratio = 1.0 / scale / scale;
    for (ssize_t j = 0; j < nrow; ++j) {
        const auto& elem = det[j];
        int y = std::get<0>(elem);
        int x = std::get<1>(elem);
        cimg.set_r(gal_array, x, y);
        cimg.fft();
        cimg.deconvolve(parr, klim);
        const py::array_t<double> meas = cimg.measure(filter_image);
        auto row = meas.unchecked<1>();
        for (ssize_t i = 0; i < ncol; ++i) {
            src_mut(j, i) = row(i) * ratio;
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
        .def(py::init<int, int, double, double, double, py::array_t<double>>(),
            "Initialize the FpfsImage object using an ndarray",
            py::arg("nx"), py::arg("ny"),
            py::arg("scale"), py::arg("sigma_arcsec"),
            py::arg("klim"),
            py::arg("psf_array")
        )
        .def("smooth_image", &FpfsImage::smooth_image,
            "Smooths the image after PSF deconvolution",
            py::arg("gal_array"),
            py::arg("noise_array")
        )
        .def("find_peaks", &FpfsImage::find_peaks,
            "Detects peaks from smoothed images",
            py::arg("gal_conv"),
            py::arg("fthres"),
            py::arg("pthres"),
            py::arg("pratio"),
            py::arg("std_m00"),
            py::arg("std_v"),
            py::arg("bound")
        )
        .def("measure_sources", &FpfsImage::measure_sources,
            "measure source properties using filter at the position of det",
            py::arg("gal_array"),
            py::arg("filter_image"),
            py::arg("det")
        );
}
}
