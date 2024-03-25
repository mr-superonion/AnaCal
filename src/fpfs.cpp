#include "anacal.h"


namespace anacal {

Fpfs::Fpfs(
    double scale,
    double sigma_arcsec,
    int nord,
    int det_nrot,
    double klim
) {
    if ((sigma_arcsec <= 0) | (sigma_arcsec > 5.0)) {
        throw std::runtime_error("Error: wrong sigma_arcsec");
    }

    this->scale = scale;
    this->sigma_arcsec = sigma_arcsec;
    this->nord = nord;
    this->det_nrot = det_nrot;
    this->klim = klim;
    sigma_f = scale / sigma_arcsec;
    const Gaussian gauss_model(sigma_f);
    return;
}

py::array_t<double>
Fpfs::smooth_image(
    const py::array_t<double>& gal_array,
    const py::array_t<double>& psf_array,
    const py::array_t<double>& noise_array
) const {
    const Gaussian gauss_model(sigma_f);

    const ssize_t* shape = gal_array.shape();
    int ny = shape[0];
    int nx = shape[1];
    Image psf_image(nx, ny, scale);
    Image gal_image(nx, ny, scale);
    // Galaxy
    gal_image.set_r(gal_array, false);
    gal_image.fft();
    // Psf
    psf_image.set_r(psf_array, true);
    psf_image.fft();
    gal_image.filter(gauss_model);
    gal_image.deconvolve(psf_image, klim);

    const ssize_t* shape_n = noise_array.shape();
    int ny_n = shape_n[0];
    int nx_n = shape_n[1];
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

Fpfs::~Fpfs() {
}


void
pyExportFpfs(py::module& m) {
    py::module_ image = m.def_submodule("image", "submodule for convolution");
    py::class_<Fpfs>(image, "Fpfs")
        .def(py::init<double, double, int, int, double>(),
            "Initialize the Fpfs object using an ndarray",
            py::arg("scale"), py::arg("sigma_arcsec"), py::arg("nord"),
            py::arg("det_nrot"), py::arg("klim")
        )
        .def("smooth_image", &Fpfs::smooth_image,
            "Sets up the image in configuration space",
            py::arg("gal_array"),
            py::arg("psf_array"),
            py::arg("noise_array")
        );
}

}
