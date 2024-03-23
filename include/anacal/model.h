#ifndef ANACAL_MODEL_H
#define ANACAL_MODEL_H

#include "stdafx.h"

namespace anacal {

    class BaseModel: public std::enable_shared_from_this<BaseModel> {
    private:
        double gamma1 = 0.0;
        double gamma2 = 0.0;
        double sin_theta = 0.0;
        double cos_theta = 1.0;

    public:
        BaseModel();
        std::tuple<double, double>
        transform(
            double kx,
            double ky
        ) const;

        virtual std::complex<double>
        fValue(double kx, double ky) const;

        std::complex<double>
        apply(
            double kx,
            double ky
        ) const;

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
        virtual ~BaseModel() = default;

    };

    class MultipliedBaseModel : public BaseModel {
    private:
        std::shared_ptr<BaseModel> f1;
        std::shared_ptr<BaseModel> f2;
    public:
        MultipliedBaseModel(
            std::shared_ptr<BaseModel> f1,
            std::shared_ptr<BaseModel> f2
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


    class DividedBaseModel : public BaseModel {
    private:
        std::shared_ptr<BaseModel> numerator;
        std::shared_ptr<BaseModel> denominator;
    public:
        DividedBaseModel(
            std::shared_ptr<BaseModel> numerator,
            std::shared_ptr<BaseModel> denominator
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
    class Gaussian : public BaseModel {
    private:
        double sigma;
        double _p0;
    public:
        Gaussian(double sigma);
        std::complex<double> fValue(double kx, double ky) const override;
    };

    /// Gaussian convolved with Tophat Function
    class GaussianTopHat : public BaseModel {
    private:
        double d, sigma;
        double _p0;
    public:
        GaussianTopHat(double d, double sigma);
        std::complex<double> fValue(double kx, double ky) const override;
    };

    void pyExportModel(py::module& m);
}
#endif
