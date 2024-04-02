#include "anacal.h"


namespace anacal {

NoiseImage::NoiseImage(
    int nx,
    int ny,
    double scale,
    double noise_std,
    bool use_estimate
): cimg(nx, ny, scale, use_estimate) {
    this->nx = nx;
    this->ny = ny;
    this->scale = scale;
    std_f = std::sqrt(nx * ny / 2.0) * noise_std;
    return;
}


NoiseImage::update_noise(
    unsigned int seed,
    const BaseModel filter_model,
) {
    std::mt19937 engine(seed);
    std::normal_distribution<double> dist(0.0, std_f);
}


py::array_t<double>&
NoiseImage::draw_r() const {
    return this->cimg.draw_r();
}

py::array_t<std::complex<double>>&
NoiseImage::draw_f() const {
    return this->cimg.draw_f();
}



NoiseImage::~NoiseImage() {
}


void
pyExportNoise(py::module& m) {
    py::module_ fpfs = m.def_submodule("fpfs", "submodule for FPFS shear estimation");
    py::class_<NoiseImage>(fpfs, "NoiseImage")
        .def(py::init<int, int, double, bool>(),
            "Initialize the NoiseImage object using an ndarray",
            py::arg("nx"), py::arg("ny"),
            py::arg("scale"), py::arg("sigma_arcsec"),
            py::arg("use_estimate")=false
        );
}
}
