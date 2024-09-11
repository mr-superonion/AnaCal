#ifndef ANACAL_MATH_POOLING_H
#define ANACAL_MATH_POOLING_H

#include <cmath>
/* #include "qnumber.h" */

namespace anacal {
namespace math {

    inline double ssfunc1(double x, double mu, double sigma) {
        // Returns the C1 smooth step weight funciton
        auto _func = [](double t) -> double {
            return -2.0 * t * t * t + 3 * t * t;
        };

        double t = (x - mu) / (sigma * 2.0) + 0.5;
        if (t < 0) return 0.0;
        else if (t <= 1) return _func(t);
        else return 1.0;
    }

    inline double ssfunc1_deriv(double x, double mu, double sigma) {
        // Returns the derivative of C1 smooth step weight funciton
        auto _func = [](double t) -> double {
            return -6.0 * t * t + 6 * t;
        };

        double t = (x - mu) / (sigma * 2.0) + 0.5;
        if (t < 0) return 0.0;
        else if (t <= 1) return _func(t) / (sigma * 2.0);
        else return 0.0;
    }

    inline double ssfunc2(double x, double mu, double sigma) {
        // Returns the C2 smooth step weight funciton
        auto _func = [](double t) -> double {
            double t3 = t * t * t;
            return (6 * t * t - 15 * t + 10) * t3 ;
        };

        double t = (x - mu) / (sigma * 2.0) + 0.5;
        if (t < 0) return 0.0;
        else if (t <= 1) return _func(t);
        else return 1.0;
    }

    inline double ssfunc2_deriv(double x, double mu, double sigma) {
        // Returns the derivative of C2 smooth step weight funciton
        auto _func = [](double t) -> double {
            double t2 = t * t;
            return (30 * t2 - 60 * t + 30) * t2 ;
        };

        double t = (x - mu) / (sigma * 2.0) + 0.5;
        if (t < 0) return 0.0;
        else if (t <= 1) return _func(t) / (sigma * 2.0);
        else return 0.0;
    }



}
}

#endif // ANACAL_MATH_POOLING_H
