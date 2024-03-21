#ifndef MODEL_H
#define MODEL_H

#include "stdafx.h"


class BaseFunc: public std::enable_shared_from_this<BaseFunc> {
private:
    double gamma1 = 0.0;
    double gamma2 = 0.0;
    double sin_theta = 0.0;
    double cos_theta = 1.0;

    std::tuple<double, double>
    _transform(
        double kx,
        double ky
    ) const {

        // Shearing
        double kx_sheared = kx * (1 - gamma1) + ky * -gamma2;
        double ky_sheared = kx * -gamma2 + ky * (1 + gamma1);

        // Rotation
        double kx_rotated = cos_theta * kx_sheared + sin_theta * ky_sheared;
        double ky_rotated = -sin_theta * kx_sheared + cos_theta * ky_sheared;
        return std::make_tuple(kx_rotated, ky_rotated);
    }
public:
    virtual std::complex<double> fValue(double kx, double ky) const {return 0;}
    std::complex<double>
    apply(
        double kx,
        double ky
    ) const {
        double kx_distorted, ky_distorted;
        std::tie(kx_distorted, ky_distorted) = _transform(kx, ky);
        return fValue(kx_distorted, ky_distorted);
    }
    // Draw Image
    py::array_t<std::complex<double>>
    draw(
        double scale,
        int nx,
        int ny
    ) const;
    void set_transform(double theta, double gamma1, double gamma2) {

        this->cos_theta = std::cos(theta);
        this->sin_theta = std::sin(theta);
        this->gamma1 = gamma1;
        this->gamma2 = gamma2;
    }
    virtual ~BaseFunc() = default;

};

class MultipliedBaseFunc : public BaseFunc {
private:
    std::shared_ptr<BaseFunc> f1;
    std::shared_ptr<BaseFunc> f2;
public:
    MultipliedBaseFunc(
        std::shared_ptr<BaseFunc> f1,
        std::shared_ptr<BaseFunc> f2
    ):
        f1(f1), f2(f2) {}

    std::complex<double>
    fValue(
        double kx,
        double ky
    ) const override {
        return f1->fValue(kx, ky) * f2->fValue(kx, ky);
    }
};


class DividedBaseFunc : public BaseFunc {
private:
    std::shared_ptr<BaseFunc> numerator;
    std::shared_ptr<BaseFunc> denominator;
public:
    DividedBaseFunc(
        std::shared_ptr<BaseFunc> numerator,
        std::shared_ptr<BaseFunc> denominator
    ):
        numerator(numerator), denominator(denominator) {}

    std::complex<double>
    fValue(
        double kx,
        double ky
    ) const override {
        std::complex<double> denomResult = denominator->fValue(kx, ky);
        return numerator->fValue(kx, ky) / denomResult;
    }

};



/// Gaussian Function
class Gaussian : public BaseFunc {
private:
    double sigma;
    double _p0;
public:
    Gaussian(double sigma);
    std::complex<double> fValue(double kx, double ky) const override;
};

/// Gaussian convolved with Tophat Function
class GaussianTopHat : public BaseFunc {
private:
    double d, sigma;
    double _p0;
public:
    GaussianTopHat(double d, double sigma);
    std::complex<double> fValue(double kx, double ky) const override;
};

#endif // MODEL_H
