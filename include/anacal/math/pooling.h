#ifndef ANACAL_MATH_POOLING_H
#define ANACAL_MATH_POOLING_H

#include <cmath>

namespace anacal {
namespace math {

    inline double ssfunc1(double x, double mu, double sigma) {
        auto _func = [](double t) -> double {
            return -2.0 * t * t * t + 3 * t * t;
        };

        double t = (x - mu) / (sigma * 2.0) + 0.5;
        if (t < 0) return 0.0;
        else if (t <= 1) return _func(t);
        else return 1.0;
    }

    inline double ssfunc2(double x, double mu, double sigma) {
        auto _func = [](double t) -> double {
            return 6 * t * t * t * t * t - 15 * t * t * t * t + 10 * t * t * t;
        };

        double t = (x - mu) / (sigma * 2.0) + 0.5;
        if (t < 0) return 0.0;
        else if (t <= 1) return _func(t);
        else return 1.0;
    }

    inline double sigmoid(double x, double mu, double sigma) {
        double t = (x - mu) / sigma;
        return 1.0 / (1.0 + std::exp(-t));
    }

}
}

#endif // ANACAL_MATH_POOLING_H
