#include "anacal.h"


namespace anacal {

BaseModel::BaseModel(){
    // Constructor implementation. Can be empty if nothing to initialize.
}
std::complex<double>
BaseModel::fValue(double, double) const {
    return 0;
}

std::pair<double, double>
BaseModel::transform(
    double kx,
    double ky
) const {

    // Shearing
    double kx_sheared = kx * (1 - gamma1) + ky * -gamma2;
    double ky_sheared = kx * -gamma2 + ky * (1 + gamma1);

    // Rotation
    double kx_rotated = cos_theta * kx_sheared + sin_theta * ky_sheared;
    double ky_rotated = -sin_theta * kx_sheared + cos_theta * ky_sheared;
    return std::make_pair(kx_rotated, ky_rotated);
}

std::complex<double>
BaseModel::apply(
    double kx,
    double ky
) const {
    std::pair<double, double> _t = transform(kx, ky);
    double kx_distorted = _t.first;
    double ky_distorted = _t.second;
    return fValue(kx_distorted, ky_distorted);
}

// Gaussian Profile
Gaussian::Gaussian(double sigma) : sigma(sigma) {
    _p0 = 1.0 / (2 * sigma * sigma);
}
std::complex<double>
Gaussian::fValue(double kx, double ky) const {
    double rpart = std::exp(-(kx*kx + ky*ky) * _p0 );
    return std::complex<double>(rpart, 0.0);
}

// Circular Tophat
CirTopHat::CirTopHat(double d) : d(d) {
    _p0 = d * d;
}
std::complex<double>
CirTopHat::fValue(double kx, double ky) const {
    double r2 = kx * kx + ky * ky;
    double rpart = r2 > _p0 ? 0 : 1;
    return std::complex<double>(rpart, 0.0);
}

// Gaussian Tophat
GaussianTopHat::GaussianTopHat(double d, double sigma) : d(d), sigma(sigma) {
    _p0 = 1.0 / (std::sqrt(2) * sigma);
}
std::complex<double>
GaussianTopHat::fValue(double kx, double ky) const {
    double factorX = std::erf((kx + d) * _p0) - std::erf((kx - d) * _p0);
    double factorY = std::erf((ky + d) * _p0) - std::erf((ky - d) * _p0);
    return std::complex<double>(factorX * factorY * 0.25, 0.0);
}

// Sqare of Gaussian Tophat
GaussianTopHat2::GaussianTopHat2(double d, double sigma) : d(d), sigma(sigma) {
    _p0 = 1.0 / (std::sqrt(2) * sigma);
}
std::complex<double>
GaussianTopHat2::fValue(double kx, double ky) const {
    double factorX = std::erf((kx + d) * _p0) - std::erf((kx - d) * _p0);
    double factorY = std::erf((ky + d) * _p0) - std::erf((ky - d) * _p0);
    double result = factorX * factorY * 0.25;
    return std::complex<double>(result * result, 0.0);
}

py::array_t<std::complex<double>>
BaseModel::draw(double scale, int nx, int ny) const {
    // Grid dimensions
    int kx_length = nx / 2 + 1;
    int ky_length = ny;

    // Prepare output array
    auto result = py::array_t<std::complex<double>>({ky_length, kx_length});
    auto r = result.mutable_unchecked<2>(); // Accessor

    double dkx = 2.0 * M_PI / nx / scale;
    double dky = 2.0 * M_PI / ny / scale;
    int ny2 = ny / 2;
    for (int iy = 0; iy < ky_length; ++iy) {
        double ky = (iy < ny2) ? iy * dky : (iy - ny) * dky ;

        for (int ix = 0; ix < kx_length; ++ix) {
            double kx = ix * dkx;
            r(iy, ix) = apply(kx, ky);
        }
    }

    return result;
}


void pyExportModel(py::module& m) {
    py::module_ model = m.def_submodule("model", "submodule for models");
    py::class_<BaseModel>(model, "BaseModel")
        .def(py::init<>())
        .def("apply", &BaseModel::apply,
            "Returns the pre-distorted function value at kx, ky",
            py::arg("kx"), py::arg("ky")
        )
        .def("fValue", &BaseModel::fValue,
            "Returns the distorted value at kx, ky",
            py::arg("kx"), py::arg("ky")
        )
        .def("set_transform", &BaseModel::set_transform,
            "Sets the transform parameters",
            py::arg("theta"), py::arg("gamma1"), py::arg("gamma2")
        )
        .def("draw", &BaseModel::draw,
            "draw the distorted function to ndarray",
            py::arg("scale"), py::arg("nx"), py::arg("ny")
        );

    py::class_<Gaussian, BaseModel>(model, "Gaussian")
        .def(py::init<double>(),
            "Gaussian basis function",
            py::arg("sigma")
        );

    py::class_<GaussianTopHat, BaseModel>(model, "GaussianTopHat")
        .def(py::init<double, double>(),
            "Gaussian convolved with top-hat basis function",
            py::arg("d"), py::arg("sigma")
        );

    py::class_<GaussianTopHat2, BaseModel>(model, "GaussianTopHat2")
        .def(py::init<double, double>(),
            "Square ofGaussian convolved with top-hat basis function",
            py::arg("d"), py::arg("sigma")
        );

    py::class_<CirTopHat, BaseModel>(model, "CirTopHat")
        .def(py::init<double>(),
            "Circular top-hat basis function",
            py::arg("d")
        );
}

}
