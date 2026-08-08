#ifndef ANACAL_MATH_POOLING_H
#define ANACAL_MATH_POOLING_H

namespace anacal {
namespace math {

    namespace detail {
        inline double scalar_of(double t) { return t; }
        inline double scalar_of(const qnumber& t) { return t.v; }
    }

    // Shared clamp shell of the smooth-step family: map x onto
    // t = (x - mu) / (2 sigma) + 1/2, return 0 below the ramp, 1 above
    // it, and the supplied polynomial on it.  Each public variant passes
    // its ORIGINAL polynomial expression, so floating-point behaviour is
    // exactly what the six hand-written copies produced.
    template <typename X, typename M, typename Poly>
    inline auto ssfunc_shell(const X& x, M mu, M sigma, Poly poly)
        -> decltype(poly((x - mu) / (sigma * 2.0) + 0.5)) {
        auto t = (x - mu) / (sigma * 2.0) + 0.5;
        using R = decltype(poly(t));
        if (detail::scalar_of(t) < 0) return R(0.0);
        else if (detail::scalar_of(t) <= 1) return poly(t);
        else return R(1.0);
    }

    // Same shell for the derivatives: zero outside the ramp on BOTH
    // sides, and the chain-rule factor 1 / (2 sigma) inside it.
    template <typename Poly>
    inline double ssfunc_deriv_shell(
        double x, double mu, double sigma, Poly poly
    ) {
        double t = (x - mu) / (sigma * 2.0) + 0.5;
        if (t < 0) return 0.0;
        else if (t <= 1) return poly(t) / (sigma * 2.0);
        else return 0.0;
    }

    // C1 smooth step
    inline qnumber ssfunc1(qnumber x, double mu, double sigma) {
        return ssfunc_shell(x, mu, sigma, [](const qnumber& t) {
            return -2.0 * pow(t, 3.0) + 3.0 * pow(t, 2);
        });
    }

    inline double ssfunc1(double x, double mu, double sigma) {
        return ssfunc_shell(x, mu, sigma, [](double t) {
            return -2.0 * t * t * t + 3 * t * t;
        });
    }

    inline double ssfunc1_deriv(double x, double mu, double sigma) {
        return ssfunc_deriv_shell(x, mu, sigma, [](double t) {
            return -6.0 * t * t + 6 * t;
        });
    }

    // C2 smooth step
    template <typename T>
    inline qnumber ssfunc2(qnumber x, T mu, T sigma) {
        return ssfunc_shell(x, mu, sigma, [](const qnumber& t) {
            return 6 * pow(t, 5) - 15 * pow(t, 4) + 10 * pow(t, 3);
        });
    }

    inline double ssfunc2(double x, double mu, double sigma) {
        return ssfunc_shell(x, mu, sigma, [](double t) {
            double t3 = t * t * t;
            return (6 * t * t - 15 * t + 10) * t3;
        });
    }

    inline double ssfunc2_deriv(double x, double mu, double sigma) {
        return ssfunc_deriv_shell(x, mu, sigma, [](double t) {
            double t2 = t * t;
            return (30 * t2 - 60 * t + 30) * t2;
        });
    }

}
}

#endif // ANACAL_MATH_POOLING_H
