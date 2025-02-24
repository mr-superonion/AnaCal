#ifndef ANACAL_MODEL_H
#define ANACAL_MODEL_H

#include "stdafx.h"

namespace anacal {
class BaseModel {
private:
    double gamma1 = 0.0;
    double gamma2 = 0.0;
    double sin_theta = 0.0;
    double cos_theta = 1.0;

    // Preventing copy (implement these if you need copy semantics)
    BaseModel(const BaseModel&) = delete;
    BaseModel& operator=(const BaseModel&) = delete;

    std::pair<double, double>
    transform(
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
    };
public:
    BaseModel() {};

    virtual std::complex<double> fValue(double, double) const {
        return 0;
    };

    std::complex<double>
    apply(
        double kx,
        double ky
    ) const {
        std::pair<double, double> _t = transform(kx, ky);
        double kx_distorted = _t.first;
        double ky_distorted = _t.second;
        return fValue(kx_distorted, ky_distorted);
    };

    // Draw Image
    py::array_t<std::complex<double>>
    draw(double scale, int nx, int ny) const {
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
    };
    void set_transform(double theta, double gamma1, double gamma2) {
        this->cos_theta = std::cos(theta);
        this->sin_theta = std::sin(theta);
        this->gamma1 = gamma1;
        this->gamma2 = gamma2;
    }


    BaseModel(BaseModel&& other) noexcept = default;
    BaseModel& operator=(BaseModel&& other) noexcept = default;

    virtual ~BaseModel() = default;

};


/// Gaussian Function
class Gaussian : public BaseModel {
private:
    double sigma;
    double _p0;
public:
    // Gaussian Profile
    Gaussian(double sigma) : sigma(sigma) {
        _p0 = 1.0 / (2 * sigma * sigma);
    };

    std::complex<double>
    fValue(double kx, double ky) const {
        double rpart = std::exp(-(kx*kx + ky*ky) * _p0 );
        return std::complex<double>(rpart, 0.0);
    };
};

/// Gaussian's g1 multiplier Function
class GaussianG1 : public BaseModel {
private:
    double sigma;
    double _p0;
public:
    GaussianG1(double sigma) : sigma(sigma) {
        _p0 = 1.0 / (sigma * sigma);
    };

    std::complex<double>
    fValue(double kx, double ky) const {
        double rpart = (kx*kx - ky*ky) * _p0;
        return std::complex<double>(rpart, 0.0);
    };
};

/// Gaussian's g2 multiplier Function
class GaussianG2 : public BaseModel {
private:
    double sigma;
    double _p0;
public:
    GaussianG2(double sigma) : sigma(sigma) {
        _p0 = 1.0 / (sigma * sigma);
    };

    std::complex<double>
    fValue(double kx, double ky) const {
        double rpart = (2.0 * kx*ky) * _p0;
        return std::complex<double>(rpart, 0.0);
    };
};

/// Gaussian's x1 multiplier Function
class GaussianX1 : public BaseModel {
private:
    double sigma;
public:
    GaussianX1() {};

    std::complex<double>
    fValue(double kx, double ky) const {
        (void)ky;
        double ipart = -kx;
        return std::complex<double>(0.0, ipart);
    };
};

/// Gaussian's x2 multiplier Function
class GaussianX2 : public BaseModel {
private:
    double sigma;
public:
    GaussianX2() {};

    std::complex<double>
    fValue(double kx, double ky) const {
        (void)kx;
        double ipart = -ky;
        return std::complex<double>(0.0, ipart);
    };
};

/// Gaussian convolved with Tophat Function
class CirTopHat : public BaseModel {
private:
    double d;
    double _p0;
public:
    // Circular Tophat
    CirTopHat(double d) : d(d) {
        _p0 = d * d;
    };

    std::complex<double>
    fValue(double kx, double ky) const {
        double r2 = kx * kx + ky * ky;
        double rpart = r2 > _p0 ? 0 : 1;
        return std::complex<double>(rpart, 0.0);
    };
};

/// Gaussian convolved with Tophat Function
class GaussianTopHat : public BaseModel {
private:
    double d, sigma;
    double _p0;
public:
    // Gaussian Tophat
    GaussianTopHat(double d, double sigma) : d(d), sigma(sigma) {
        _p0 = 1.0 / (std::sqrt(2) * sigma);
    };

    std::complex<double>
    fValue(double kx, double ky) const {
        double factorX = std::erf((kx + d) * _p0) - std::erf((kx - d) * _p0);
        double factorY = std::erf((ky + d) * _p0) - std::erf((ky - d) * _p0);
        return std::complex<double>(factorX * factorY * 0.25, 0.0);
    };
};

/// Gaussian convolved with Tophat Function
class GaussianTopHat2 : public BaseModel {
private:
    double d, sigma;
    double _p0;
public:
    // Sqare of Gaussian Tophat
    GaussianTopHat2(double d, double sigma) : d(d), sigma(sigma) {
        _p0 = 1.0 / (std::sqrt(2) * sigma);
    };

    std::complex<double>
    fValue(double kx, double ky) const {
        double factorX = std::erf((kx + d) * _p0) - std::erf((kx - d) * _p0);
        double factorY = std::erf((ky + d) * _p0) - std::erf((ky - d) * _p0);
        double result = factorX * factorY * 0.25;
        return std::complex<double>(result * result, 0.0);
    };
};

void pyExportModel(py::module& m);
}
#endif // ANACAL_MODEL_H
