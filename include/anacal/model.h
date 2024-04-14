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

        std::pair<double, double> transform(double, double) const;
    public:
        BaseModel();

        virtual std::complex<double> fValue(double, double) const;

        std::complex<double> apply(double, double) const;

        // Draw Image
        py::array_t<std::complex<double>>
        draw(double, int, int) const;
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
        Gaussian(double);
        std::complex<double> fValue(double, double) const override;
    };

    /// Gaussian convolved with Tophat Function
    class CirTopHat : public BaseModel {
    private:
        double d;
        double _p0;
    public:
        CirTopHat(double);
        std::complex<double> fValue(double, double) const override;
    };

    /// Gaussian convolved with Tophat Function
    class GaussianTopHat : public BaseModel {
    private:
        double d, sigma;
        double _p0;
    public:
        GaussianTopHat(double, double);
        std::complex<double> fValue(double, double) const override;
    };

    /// Gaussian convolved with Tophat Function
    class GaussianTopHat2 : public BaseModel {
    private:
        double d, sigma;
        double _p0;
    public:
        GaussianTopHat2(double, double);
        std::complex<double> fValue(double, double) const override;
    };

    void pyExportModel(py::module& m);
}
#endif // ANACAL_MODEL_H
