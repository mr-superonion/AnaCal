#include "anacal.h"


namespace anacal {

FpfsDetect::FpfsDetect(
    double scale,
    double sigma_arcsec,
    int det_nrot,
    double klim
) {
    if ((sigma_arcsec <= 0) || (sigma_arcsec > 5.0)) {
        throw std::runtime_error("Error: wrong sigma_arcsec");
    }

    this->scale = scale;
    this->sigma_arcsec = sigma_arcsec;
    this->det_nrot = det_nrot;
    this->klim = klim;
    sigma_f = 1.0 / sigma_arcsec;
    return;
}

py::array_t<double>
FpfsDetect::smooth_image(
    const py::array_t<double>& gal_array,
    const py::array_t<double>& psf_array,
    const py::array_t<double>& noise_array
) const {
    const Gaussian gauss_model(sigma_f);

    const ssize_t* shape = gal_array.shape();
    int ny = shape[0];
    int nx = shape[1];
    // Galaxy
    Image gal_image(nx, ny, scale);
    gal_image.set_r(gal_array, false);
    gal_image.fft();
    // Psf
    Image psf_image(nx, ny, scale);
    psf_image.set_r(psf_array, true);
    psf_image.fft();

    gal_image.filter(gauss_model);
    gal_image.deconvolve(psf_image, klim);

    const ssize_t* shape_n = noise_array.shape();
    ssize_t ny_n = shape_n[0];
    ssize_t nx_n = shape_n[1];
    if ((ny_n == ny) & (nx_n == nx)) {
        Image noise_image(nx, ny, scale);
        noise_image.set_r(noise_array, false);
        noise_image.fft();
        psf_image.rotate90_f();
        noise_image.filter(gauss_model);
        noise_image.deconvolve(psf_image, klim);
        gal_image.add_image_f(noise_image);
    }

    gal_image.ifft();
    py::array_t<double> gal_conv = gal_image.draw_r();
    return gal_conv;
}


std::vector<std::tuple<int, int, bool>>
FpfsDetect::detect_peaks(
    const py::array_t<double>& gal_conv,
    double fthres,
    double pthres,
    double pratio,
    double std_m00,
    double std_v,
    int bound
) const {
    if (gal_conv.ndim() != 2)
        throw std::runtime_error("Error: Input should be a 2D array");
    auto r = gal_conv.unchecked<2>();
    ssize_t ny = r.shape(0);
    ssize_t nx = r.shape(1);

    double fcut = fthres * std_m00;
    double pcut = pthres * std_v;

    std::vector<std::tuple<int, int, bool>> peaks;
    for (ssize_t j = bound + 1; j < ny - bound; ++j) {
        for (ssize_t i = bound + 1; i < nx - bound; ++i) {
            double c = r(j, i);
            bool sel = (
                (c > fcut) &&
                (c > r(j-1, i)) &&
                (c > r(j+1, i)) &&
                (c > r(j, i-1)) &&
                (c > r(j, i+1))
            );
            double thres = pcut + pratio * c;
            if (sel) {
                bool is_peak = (
                    (c > r(i-1, j) - thres) &&
                    (c > r(i+1, j) - thres) &&
                    (c > r(i, j-1) - thres) &&
                    (c > r(i, j+1) - thres)
                );
                peaks.push_back({j, i, is_peak});
            }
        }
    }
    return peaks;
}

FpfsDetect::~FpfsDetect() {
}


void
pyExportFpfs(py::module& m) {
    py::module_ fpfs = m.def_submodule("fpfs", "submodule for FPFS shear estimation");
    py::class_<FpfsDetect>(fpfs, "FpfsDetect")
        .def(py::init<double, double, int, double>(),
            "Initialize the FpfsDetect object using an ndarray",
            py::arg("scale"), py::arg("sigma_arcsec"),
            py::arg("det_nrot"), py::arg("klim")
        )
        .def("smooth_image", &FpfsDetect::smooth_image,
            "Smooths the image after PSF deconvolution",
            py::arg("gal_array"),
            py::arg("psf_array"),
            py::arg("noise_array")
        )
        .def("detect_peaks", &FpfsDetect::detect_peaks,
            "Detects peaks from smoothed images",
            py::arg("gal_conv"),
            py::arg("fthres"),
            py::arg("pthres"),
            py::arg("pratio"),
            py::arg("std_m00"),
            py::arg("std_v"),
            py::arg("bound")
        );
}

}
