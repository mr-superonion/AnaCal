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
        // Fields from FpfsShapelets
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


    inline FpfsShapeletsResponse calculate_shapelets_dg(
        const FpfsShapelets& x
    ) {
        double m00_g1 = -std::sqrt(2.0) * x.m22c;
        double m00_g2 = -std::sqrt(2.0) * x.m22s;
        double m20_g1 = -std::sqrt(6.0) * x.m42c;
        double m20_g2 = -std::sqrt(6.0) * x.m42s;

        double m22c_g1 = (
            1.0 / std::sqrt(2.0)
        ) * (x.m00 - x.m40) - std::sqrt(3.0) * x.m44c;
        double m22s_g2 = (
            1.0 / std::sqrt(2.0)
        ) * (x.m00 - x.m40) + std::sqrt(3.0) * x.m44c;

        // Off-diagonal terms
        // double m22c_g2 = -std::sqrt(3.0) * x.m44s;
        // double m22s_g1 = -std::sqrt(3.0) * x.m44s;

        double m42c_g1 = (
            std::sqrt(6.0) / 2.0
        ) * (x.m20 - x.m60) - std::sqrt(5.0) * x.m64c;
        double m42s_g2 = (
            std::sqrt(6.0) / 2.0
        ) * (x.m20 - x.m60) + std::sqrt(5.0) * x.m64c;

        // Off-diagonal terms
        // double m42c_g2 = -std::sqrt(5.0) * x.m64s;
        // double m42s_g1 = -std::sqrt(5.0) * x.m64s;

        return FpfsShapeletsResponse{
            m00_g1, m00_g2, m20_g1,
            m20_g2, m22c_g1, m22s_g2,
            m42c_g1, m42s_g2
        };
    };

    inline FpfsShapeletsResponse calculate_shapelets_dg(
        const FpfsDetect& x
    ) {
        double m00_g1 = -std::sqrt(2.0) * x.m22c;
        double m00_g2 = -std::sqrt(2.0) * x.m22s;
        double m20_g1 = -std::sqrt(6.0) * x.m42c;
        double m20_g2 = -std::sqrt(6.0) * x.m42s;

        double m22c_g1 = (
            1.0 / std::sqrt(2.0)
        ) * (x.m00 - x.m40) - std::sqrt(3.0) * x.m44c;
        double m22s_g2 = (
            1.0 / std::sqrt(2.0)
        ) * (x.m00 - x.m40) + std::sqrt(3.0) * x.m44c;

        // Off-diagonal terms
        // double m22c_g2 = -std::sqrt(3.0) * x.m44s;
        // double m22s_g1 = -std::sqrt(3.0) * x.m44s;

        double m42c_g1 = (
            std::sqrt(6.0) / 2.0
        ) * (x.m20 - x.m60) - std::sqrt(5.0) * x.m64c;
        double m42s_g2 = (
            std::sqrt(6.0) / 2.0
        ) * (x.m20 - x.m60) + std::sqrt(5.0) * x.m64c;

        // Off-diagonal terms
        // double m42c_g2 = -std::sqrt(5.0) * x.m64s;
        // double m42s_g1 = -std::sqrt(5.0) * x.m64s;

        return FpfsShapeletsResponse{
            m00_g1, m00_g2, m20_g1,
            m20_g2, m22c_g1, m22s_g2,
            m42c_g1, m42s_g2
        };
    };

    inline FpfsShape calculate_fpfs_ell(
        const FpfsShapelets& x, const FpfsShapeletsResponse& x_dg, double C0
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

    inline FpfsShape calculate_fpfs_ell(
        const FpfsDetect& x, const FpfsShapeletsResponse& x_dg, double C0
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


    inline FpfsWeight calculate_wsel(
        const FpfsShapelets &x,
        const FpfsShapeletsResponse &response,
        double m00_min,
        double sigma_m00,
        double r2_min,
        double sigma_r2
    ) {
        // Selection on flux
        double w0l = math::ssfunc2(x.m00, m00_min, sigma_m00);
        double dw0l = math::ssfunc2_deriv(x.m00, m00_min, sigma_m00);
        double w0l_g1 = dw0l * response.m00_g1;
        double w0l_g2 = dw0l * response.m00_g2;

        double w0u = math::ssfunc2(-x.m00, -500, sigma_m00);
        double dw0u = math::ssfunc2_deriv(-x.m00, -500, sigma_m00);
        double w0u_g1 = dw0u * -response.m00_g1;
        double w0u_g2 = dw0u * -response.m00_g2;

        // Selection on size (lower limit)
        // (M00 + M20) / M00 > r2_min
        double r2l = x.m00 * (1.0 - r2_min) + x.m20;
        double w2l = math::ssfunc2(r2l, sigma_r2, sigma_r2);
        double dw2l = math::ssfunc2_deriv(r2l, sigma_r2, sigma_r2);
        double w2l_g1 = dw2l * (
            response.m00_g1 * (1.0 - r2_min) + response.m20_g1
        );
        double w2l_g2 = dw2l * (
            response.m00_g2 * (1.0 - r2_min) + response.m20_g2
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

    inline FpfsWeight calculate_wsel(
        const FpfsDetect &x,
        const FpfsShapeletsResponse &response,
        double m00_min,
        double sigma_m00,
        double r2_min,
        double sigma_r2
    ) {
        // Selection on flux
        double w0l = math::ssfunc2(x.m00, m00_min, sigma_m00);
        double dw0l = math::ssfunc2_deriv(x.m00, m00_min, sigma_m00);
        double w0l_g1 = dw0l * response.m00_g1;
        double w0l_g2 = dw0l * response.m00_g2;

        double w0u = math::ssfunc2(-x.m00, -500, sigma_m00);
        double dw0u = math::ssfunc2_deriv(-x.m00, -500, sigma_m00);
        double w0u_g1 = dw0u * -response.m00_g1;
        double w0u_g2 = dw0u * -response.m00_g2;

        // Selection on size (lower limit)
        // (M00 + M20) / M00 > r2_min
        double r2l = x.m00 * (1.0 - r2_min) + x.m20;
        double w2l = math::ssfunc2(r2l, sigma_r2, sigma_r2);
        double dw2l = math::ssfunc2_deriv(r2l, sigma_r2, sigma_r2);
        double w2l_g1 = dw2l * (
            response.m00_g1 * (1.0 - r2_min) + response.m20_g1
        );
        double w2l_g2 = dw2l * (
            response.m00_g2 * (1.0 - r2_min) + response.m20_g2
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
        const FpfsDetect &x,
        const FpfsDetect &y,
        double sigma_v,
        double pcut,
        double pthres
    ) {
        // det0 computation
        double det0 = math::ssfunc2(x.v0, sigma_v - pcut, sigma_v);
        double det0_deriv = math::ssfunc2_deriv(x.v0, sigma_v - pcut, sigma_v);
        double det0_g1 = det0_deriv * (x.v0_g1 - 2.0 * y.v0_g1);
        double det0_g2 = det0_deriv * (x.v0_g2 - 2.0 * y.v0_g2);

        // det1 computation
        double det1 = math::ssfunc2(x.v1, sigma_v - pcut, sigma_v);
        double det1_deriv = math::ssfunc2_deriv(x.v1, sigma_v - pcut, sigma_v);
        double det1_g1 = det1_deriv * (x.v1_g1 - 2.0 * y.v1_g1);
        double det1_g2 = det1_deriv * (x.v1_g2 - 2.0 * y.v1_g2);

        // det2 computation
        double det2 = math::ssfunc2(x.v2, sigma_v - pcut, sigma_v);
        double det2_deriv = math::ssfunc2_deriv(x.v2, sigma_v - pcut, sigma_v);
        double det2_g1 = det2_deriv * (x.v2_g1 - 2.0 * y.v2_g1);
        double det2_g2 = det2_deriv * (x.v2_g2 - 2.0 * y.v2_g2);

        // det3 computation
        double det3 = math::ssfunc2(x.v3, sigma_v - pcut, sigma_v);
        double det3_deriv = math::ssfunc2_deriv(x.v3, sigma_v - pcut, sigma_v);
        double det3_g1 = det3_deriv * (x.v3_g1 - 2.0 * y.v3_g1);
        double det3_g2 = det3_deriv * (x.v3_g2 - 2.0 * y.v3_g2);

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


    void pybindFpfsCatalog(py::module_& fpfs);
}

#endif // ANACAL_FPFS_CAT_H
