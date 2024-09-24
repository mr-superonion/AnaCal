#ifndef ANACAL_FPFS_CAT_H
#define ANACAL_FPFS_CAT_H


#include "defaults.h"

namespace anacal {
    struct FpfsShapelets {
        double m00;
        double m20;
        double m22c;
        double m22s;
        double m40;
        double m42c;
        double m42s;
        double m44c;
        double m44s;
        double m60;
        double m64c;
        double m64s;


        // Define the addition operator
        FpfsShapelets operator+(const FpfsShapelets &other) const {
            return FpfsShapelets{
                m00 + other.m00,
                m20 + other.m20,
                m22c + other.m22c,
                m22s + other.m22s,
                m40 + other.m40,
                m42c + other.m42c,
                m42s + other.m42s,
                m44c + other.m44c,
                m44s + other.m44s,
                m60 + other.m60,
                m64c + other.m64c,
                m64s + other.m64s,
            };
        };

        // Define the subtraction operator
        FpfsShapelets operator-(const FpfsShapelets &other) const {
            return FpfsShapelets{
                m00 - other.m00,
                m20 - other.m20,
                m22c - other.m22c,
                m22s - other.m22s,
                m40 - other.m40,
                m42c - other.m42c,
                m42s - other.m42s,
                m44c - other.m44c,
                m44s - other.m44s,
                m60 - other.m60,
                m64c - other.m64c,
                m64s - other.m64s,
            };
        };

        // Define multiplication by a scalar from the right-hand side
        // (FpfsShapelets * scalar)
        template <typename T>
        FpfsShapelets operator*(const T &scalar) const {
            return FpfsShapelets{
                m00 * scalar,
                m20 * scalar,
                m22c * scalar,
                m22s * scalar,
                m40 * scalar,
                m42c * scalar,
                m42s * scalar,
                m44c * scalar,
                m44s * scalar,
                m60 * scalar,
                m64c * scalar,
                m64s * scalar,
            };
        };
    };


    struct FpfsShapeletsResponse {
        double m00_g1;
        double m00_g2;
        double m20_g1;
        double m20_g2;
        double m22c_g1;
        double m22s_g2;
        double m42c_g1;
        double m42s_g2;
    };

    struct FpfsDetect {
        // Fields from FpfsDetect
        double m00;
        double m20;
        double m22c;
        double m22s;
        double m40;
        double m42c;
        double m42s;
        double m44c;
        double m44s;
        double m60;
        double m64c;
        double m64s;
        double v0;
        double v1;
        double v2;
        double v3;
        double v0_g1;
        double v1_g1;
        double v2_g1;
        double v3_g1;
        double v0_g2;
        double v1_g2;
        double v2_g2;
        double v3_g2;

        // Define the addition operator
        FpfsDetect operator+(const FpfsDetect &other) const {
            return FpfsDetect{
                m00 + other.m00,
                m20 + other.m20,
                m22c + other.m22c,
                m22s + other.m22s,
                m40 + other.m40,
                m42c + other.m42c,
                m42s + other.m42s,
                m44c + other.m44c,
                m44s + other.m44s,
                m60 + other.m60,
                m64c + other.m64c,
                m64s + other.m64s,
                v0 + other.v0,
                v1 + other.v1,
                v2 + other.v2,
                v3 + other.v3,
                v0_g1 + other.v0_g1,
                v1_g1 + other.v1_g1,
                v2_g1 + other.v2_g1,
                v3_g1 + other.v3_g1,
                v0_g2 + other.v0_g2,
                v1_g2 + other.v1_g2,
                v2_g2 + other.v2_g2,
                v3_g2 + other.v3_g2,
            };
        };

        // Define the subtraction operator
        FpfsDetect operator-(const FpfsDetect &other) const {
            return FpfsDetect{
                m00 - other.m00,
                m20 - other.m20,
                m22c - other.m22c,
                m22s - other.m22s,
                m40 - other.m40,
                m42c - other.m42c,
                m42s - other.m42s,
                m44c - other.m44c,
                m44s - other.m44s,
                m60 - other.m60,
                m64c - other.m64c,
                m64s - other.m64s,
                v0 - other.v0,
                v1 - other.v1,
                v2 - other.v2,
                v3 - other.v3,
                v0_g1 - other.v0_g1,
                v1_g1 - other.v1_g1,
                v2_g1 - other.v2_g1,
                v3_g1 - other.v3_g1,
                v0_g2 - other.v0_g2,
                v1_g2 - other.v1_g2,
                v2_g2 - other.v2_g2,
                v3_g2 - other.v3_g2,
            };
        };

        // Define multiplication by a scalar from the right-hand side
        // (FpfsDetect * scalar)
        template <typename T>
        FpfsDetect operator*(const T &scalar) const {
            return FpfsDetect{
                m00 * scalar,
                m20 * scalar,
                m22c * scalar,
                m22s * scalar,
                m40 * scalar,
                m42c * scalar,
                m42s * scalar,
                m44c * scalar,
                m44s * scalar,
                m60 * scalar,
                m64c * scalar,
                m64s * scalar,
                v0 * scalar,
                v1 * scalar,
                v2 * scalar,
                v3 * scalar,
                v0_g1 * scalar,
                v1_g1 * scalar,
                v2_g1 * scalar,
                v3_g1 * scalar,
                v0_g2 * scalar,
                v1_g2 * scalar,
                v2_g2 * scalar,
                v3_g2 * scalar,
            };
        };
    };


    struct FpfsShape {
        double e1;
        double e1_g1;
        double e2;
        double e2_g2;
        double q1;
        double q1_g1;
        double q2;
        double q2_g2;
    };

    struct FpfsWeight {
        double w;
        double w_g1;
        double w_g2;
    };

    struct FpfsCatalog {
        double e1;
        double e1_g1;
        double e2;
        double e2_g2;
        double q1;
        double q1_g1;
        double q2;
        double q2_g2;
        double w;
        double w_g1;
        double w_g2;
    };


    template <typename T>
    inline FpfsShapeletsResponse calculate_shapelets_dg(
        const T& x,
        const std::optional<T>& y
    ) {
        T xx = y.has_value() ? x - *y * 2.0 : x;
        double m00_g1 = -std::sqrt(2.0) * xx.m22c;
        double m00_g2 = -std::sqrt(2.0) * xx.m22s;
        double m20_g1 = -std::sqrt(6.0) * xx.m42c;
        double m20_g2 = -std::sqrt(6.0) * xx.m42s;

        double m22c_g1 = (
            1.0 / std::sqrt(2.0)
        ) * (xx.m00 - xx.m40) - std::sqrt(3.0) * xx.m44c;
        double m22s_g2 = (
            1.0 / std::sqrt(2.0)
        ) * (xx.m00 - xx.m40) + std::sqrt(3.0) * xx.m44c;

        // Off-diagonal terms
        // double m22c_g2 = -std::sqrt(3.0) * xx.m44s;
        // double m22s_g1 = -std::sqrt(3.0) * xx.m44s;

        double m42c_g1 = (
            std::sqrt(6.0) / 2.0
        ) * (xx.m20 - xx.m60) - std::sqrt(5.0) * xx.m64c;
        double m42s_g2 = (
            std::sqrt(6.0) / 2.0
        ) * (xx.m20 - xx.m60) + std::sqrt(5.0) * xx.m64c;

        // Off-diagonal terms
        // double m42c_g2 = -std::sqrt(5.0) * xx.m64s;
        // double m42s_g1 = -std::sqrt(5.0) * xx.m64s;

        return FpfsShapeletsResponse{
            m00_g1, m00_g2, m20_g1,
            m20_g2, m22c_g1, m22s_g2,
            m42c_g1, m42s_g2
        };
    };

    template <typename T>
    inline FpfsShape calculate_fpfs_ell(
        double C0,
        const T& x,
        const FpfsShapeletsResponse& x_dg
    ) {
        // Denominator
        double denom = x.m00 + C0;

        // Compute ellipticity 1
        double e1 = x.m22c / denom;
        double e1_g1 = x_dg.m22c_g1 / denom
            - (x_dg.m00_g1 * x.m22c) / (denom * denom);

        // Compute ellipticity 2
        double e2 = x.m22s / denom;
        double e2_g2 = x_dg.m22s_g2 / denom
            - (x_dg.m00_g2 * x.m22s) / (denom * denom);

        // Compute ellipticity 1 (4th order)
        double q1 = x.m42c / denom;
        double q1_g1 = x_dg.m42c_g1 / denom
            - (x_dg.m00_g1 * x.m42c) / (denom * denom);

        // Compute ellipticity 2 (4th order)
        double q2 = x.m42s / denom;
        double q2_g2 = x_dg.m42s_g2 / denom
            - (x_dg.m00_g2 * x.m42s) / (denom * denom);

        // Return the result as FpfsShape
        return FpfsShape{
            e1, e1_g1, e2, e2_g2, q1, q1_g1, q2, q2_g2
        };
    }

    template <typename T>
    inline FpfsWeight calculate_wsel(
        double m00_min,
        double sigma_m00,
        double r2_min,
        double sigma_r2,
        const T &x,
        const FpfsShapeletsResponse &x_dg
    ) {
        // Selection on flux
        double w0l = math::ssfunc2(x.m00, m00_min, sigma_m00);
        double dw0l = math::ssfunc2_deriv(x.m00, m00_min, sigma_m00);
        double w0l_g1 = dw0l * x_dg.m00_g1;
        double w0l_g2 = dw0l * x_dg.m00_g2;

        double w0u = math::ssfunc2(-x.m00, -500, sigma_m00);
        double dw0u = math::ssfunc2_deriv(-x.m00, -500, sigma_m00);
        double w0u_g1 = dw0u * -x_dg.m00_g1;
        double w0u_g2 = dw0u * -x_dg.m00_g2;

        // Selection on size (lower limit)
        // (M00 + M20) / M00 > r2_min
        double r2l = x.m00 * (1.0 - r2_min) + x.m20;
        double w2l = math::ssfunc2(r2l, sigma_r2, sigma_r2);
        double dw2l = math::ssfunc2_deriv(r2l, sigma_r2, sigma_r2);
        double w2l_g1 = dw2l * (
            x_dg.m00_g1 * (1.0 - r2_min) + x_dg.m20_g1
        );
        double w2l_g2 = dw2l * (
            x_dg.m00_g2 * (1.0 - r2_min) + x_dg.m20_g2
        );

        // Compute the selection weights
        double wsel = w0l * w0u * w2l;
        double wsel_g1 = (
            w0l_g1 * w0u * w2l + w0l * w0u_g1 * w2l + w0l * w0u * w2l_g1
        );
        double wsel_g2 = (
            w0l_g2 * w0u * w2l + w0l * w0u_g2 * w2l + w0l * w0u * w2l_g2
        );

        return FpfsWeight{
            wsel, wsel_g1, wsel_g2
        };
    };

    inline FpfsWeight calculate_wdet(
        double sigma_v,
        double pcut,
        double pthres,
        const FpfsDetect &x,
        const std::optional<FpfsDetect> &y
    ) {

        FpfsDetect xx = y.has_value() ? x - *y * 2.0 : x;
        // det0 computation
        double det0 = math::ssfunc2(x.v0, sigma_v - pcut, sigma_v);
        double det0_deriv = math::ssfunc2_deriv(x.v0, sigma_v - pcut, sigma_v);
        double det0_g1 = det0_deriv * (xx.v0_g1);
        double det0_g2 = det0_deriv * (xx.v0_g2);

        // det1 computation
        double det1 = math::ssfunc2(x.v1, sigma_v - pcut, sigma_v);
        double det1_deriv = math::ssfunc2_deriv(x.v1, sigma_v - pcut, sigma_v);
        double det1_g1 = det1_deriv * (xx.v1_g1);
        double det1_g2 = det1_deriv * (xx.v1_g2);

        // det2 computation
        double det2 = math::ssfunc2(x.v2, sigma_v - pcut, sigma_v);
        double det2_deriv = math::ssfunc2_deriv(x.v2, sigma_v - pcut, sigma_v);
        double det2_g1 = det2_deriv * (xx.v2_g1);
        double det2_g2 = det2_deriv * (xx.v2_g2);

        // det3 computation
        double det3 = math::ssfunc2(x.v3, sigma_v - pcut, sigma_v);
        double det3_deriv = math::ssfunc2_deriv(x.v3, sigma_v - pcut, sigma_v);
        double det3_g1 = det3_deriv * (xx.v3_g1);
        double det3_g2 = det3_deriv * (xx.v3_g2);

        // Compute the weights
        double w = det0 * det1 * det2 * det3;
        double w_g1 = (
            det0_g1 * det1 * det2 * det3 + det0 * det1_g1 * det2 * det3
        ) + (det0 * det1 * det2_g1 * det3 + det0 * det1 * det2 * det3_g1);
        double w_g2 = (
            det0_g2 * det1 * det2 * det3 + det0 * det1_g2 * det2 * det3
        ) + (det0 * det1 * det2_g2 * det3 + det0 * det1 * det2 * det3_g2);

        // Final selection based on w
        double wdet = math::ssfunc2(w, pthres, fpfs_det_sigma2);
        double wdet_deriv = math::ssfunc2_deriv(w, pthres, fpfs_det_sigma2);

        return FpfsWeight{
            wdet, wdet_deriv * w_g1, wdet_deriv * w_g2
        };
    };

    inline FpfsCatalog m2e(
        double C0,
        const FpfsShapelets &x,
        const std::optional<FpfsShapelets> &y
    ){
        FpfsShapeletsResponse x_dg = calculate_shapelets_dg(
            x, y
        );
        FpfsShape ell = calculate_fpfs_ell(
            C0, x, x_dg
        );

    };


    void pybindFpfsCatalog(py::module_& fpfs);
}

#endif // ANACAL_FPFS_CAT_H
