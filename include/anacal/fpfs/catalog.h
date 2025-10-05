#ifndef ANACAL_FPFS_CAT_H
#define ANACAL_FPFS_CAT_H


#include "base.h"

namespace anacal {
    namespace fpfs {
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

        // Define left multiplication by a scalar (scalar * FpfsShapelets)
        template <typename T>
        friend FpfsShapelets operator*(const T &scalar, const FpfsShapelets &shapelets) {
            return FpfsShapelets{
                shapelets.m00 * scalar,
                shapelets.m20 * scalar,
                shapelets.m22c * scalar,
                shapelets.m22s * scalar,
                shapelets.m40 * scalar,
                shapelets.m42c * scalar,
                shapelets.m42s * scalar,
                shapelets.m44c * scalar,
                shapelets.m44s * scalar,
                shapelets.m60 * scalar,
                shapelets.m64c * scalar,
                shapelets.m64s * scalar,
            };
        };

        // Define division by a scalar (FpfsShapelets / scalar)
        template <typename T>
        FpfsShapelets operator/(const T &scalar) const {
            return FpfsShapelets{
                m00 / scalar,
                m20 / scalar,
                m22c / scalar,
                m22s / scalar,
                m40 / scalar,
                m42c / scalar,
                m42s / scalar,
                m44c / scalar,
                m44s / scalar,
                m60 / scalar,
                m64c / scalar,
                m64s / scalar,
            };
        };
    };


        struct FpfsShapeletsResponse {
        double dm00_dg1;
        double dm00_dg2;
        double dm20_dg1;
        double dm20_dg2;
        double dm22c_dg1;
        double dm22s_dg2;
        double dm42c_dg1;
        double dm42s_dg2;
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
        double dv0_dg1;
        double dv1_dg1;
        double dv2_dg1;
        double dv3_dg1;
        double dv0_dg2;
        double dv1_dg2;
        double dv2_dg2;
        double dv3_dg2;

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
                dv0_dg1 + other.dv0_dg1,
                dv1_dg1 + other.dv1_dg1,
                dv2_dg1 + other.dv2_dg1,
                dv3_dg1 + other.dv3_dg1,
                dv0_dg2 + other.dv0_dg2,
                dv1_dg2 + other.dv1_dg2,
                dv2_dg2 + other.dv2_dg2,
                dv3_dg2 + other.dv3_dg2,
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
                dv0_dg1 - other.dv0_dg1,
                dv1_dg1 - other.dv1_dg1,
                dv2_dg1 - other.dv2_dg1,
                dv3_dg1 - other.dv3_dg1,
                dv0_dg2 - other.dv0_dg2,
                dv1_dg2 - other.dv1_dg2,
                dv2_dg2 - other.dv2_dg2,
                dv3_dg2 - other.dv3_dg2,
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
                dv0_dg1 * scalar,
                dv1_dg1 * scalar,
                dv2_dg1 * scalar,
                dv3_dg1 * scalar,
                dv0_dg2 * scalar,
                dv1_dg2 * scalar,
                dv2_dg2 * scalar,
                dv3_dg2 * scalar,
            };
        };
    };

        struct FpfsShape {
        double e1;
        double de1_dg1;
        double e2;
        double de2_dg2;
        double q1;
        double dq1_dg1;
        double q2;
        double dq2_dg2;
        double m00;
        double dm00_dg1;
        double dm00_dg2;
        double m20;
        double dm20_dg1;
        double dm20_dg2;
        double m22c;
        double dm22c_dg1;
        double m22s;
        double dm22s_dg2;
    };

        struct FpfsWeight {
        double w;
        double dw_dg1;
        double dw_dg2;
    };

        struct FpfsCatalog {
        double e1;
        double de1_dg1;
        double e2;
        double de2_dg2;
        double q1;
        double dq1_dg1;
        double q2;
        double dq2_dg2;
        double w;
        double dw_dg1;
        double dw_dg2;
        double m00;
        double dm00_dg1;
        double dm00_dg2;
        double m20;
        double dm20_dg1;
        double dm20_dg2;
    };


        template <typename T>
        inline FpfsShapeletsResponse measure_shapelets_dg(
        const T& x,
        const std::optional<T>& y=std::nullopt
    ) {
        T xx = y.has_value() ? x - *y * 2.0 : x;
        double dm00_dg1 = -std::sqrt(2.0) * xx.m22c;
        double dm00_dg2 = -std::sqrt(2.0) * xx.m22s;
        double dm20_dg1 = -std::sqrt(6.0) * xx.m42c;
        double dm20_dg2 = -std::sqrt(6.0) * xx.m42s;

        double dm22c_dg1 = (
            1.0 / std::sqrt(2.0)
        ) * (xx.m00 - xx.m40) - std::sqrt(3.0) * xx.m44c;
        double dm22s_dg2 = (
            1.0 / std::sqrt(2.0)
        ) * (xx.m00 - xx.m40) + std::sqrt(3.0) * xx.m44c;

        // Off-diagonal terms
        // double dm22c_dg2 = -std::sqrt(3.0) * xx.m44s;
        // double dm22s_dg1 = -std::sqrt(3.0) * xx.m44s;

        double dm42c_dg1 = (
            std::sqrt(6.0) / 2.0
        ) * (xx.m20 - xx.m60) - std::sqrt(5.0) * xx.m64c;
        double dm42s_dg2 = (
            std::sqrt(6.0) / 2.0
        ) * (xx.m20 - xx.m60) + std::sqrt(5.0) * xx.m64c;

        // Off-diagonal terms
        // double dm42c_dg2 = -std::sqrt(5.0) * xx.m64s;
        // double dm42s_dg1 = -std::sqrt(5.0) * xx.m64s;

        return FpfsShapeletsResponse{
            dm00_dg1, dm00_dg2, dm20_dg1,
            dm20_dg2, dm22c_dg1, dm22s_dg2,
            dm42c_dg1, dm42s_dg2
        };
    };

        template <typename T>
        inline py::array_t<FpfsShapeletsResponse> measure_shapelets_dg(
        const py::array_t<T> &x_array,
        const std::optional<py::array_t<T>> &y_array=std::nullopt
    ) {
        auto x_r = x_array.template unchecked<1>();
        int nn = x_array.shape(0);
        py::array_t<FpfsShapeletsResponse> out(nn);
        auto out_r = out.template mutable_unchecked<1>();
        if (y_array.has_value()) {
            auto y_r = y_array->template unchecked<1>();
            for (ssize_t i = 0; i < nn; ++i) {
                out_r(i) = measure_shapelets_dg<T>(
                    x_r(i),
                    y_r(i)
                );
            }
        } else {
            for (ssize_t i = 0; i < nn; ++i) {
                out_r(i) = measure_shapelets_dg<T>(
                    x_r(i)
                );
            }
        }
        return out;
    };

        template <typename T>
        inline FpfsShape measure_fpfs_shape(
        double C0,
        const T& x,
        const FpfsShapeletsResponse& x_dg
    ) {
        // Denominator
        double denom = x.m00 + C0;

        // Compute ellipticity 1
        double e1 = x.m22c / denom;
        double e1_dg1 = x_dg.dm22c_dg1 / denom
            - (x_dg.dm00_dg1 * x.m22c) / (denom * denom);

        // Compute ellipticity 2
        double e2 = x.m22s / denom;
        double e2_dg2 = x_dg.dm22s_dg2 / denom
            - (x_dg.dm00_dg2 * x.m22s) / (denom * denom);

        // Compute ellipticity 1 (4th order)
        double q1 = x.m42c / denom;
        double q1_dg1 = x_dg.dm42c_dg1 / denom
            - (x_dg.dm00_dg1 * x.m42c) / (denom * denom);

        // Compute ellipticity 2 (4th order)
        double q2 = x.m42s / denom;
        double q2_dg2 = x_dg.dm42s_dg2 / denom
            - (x_dg.dm00_dg2 * x.m42s) / (denom * denom);

        // Return the result as FpfsShape
        return FpfsShape{
            e1, e1_dg1, e2, e2_dg2,
            q1, q1_dg1, q2, q2_dg2,
            x.m00, x_dg.dm00_dg1, x_dg.dm00_dg2,
            x.m20, x_dg.dm20_dg1, x_dg.dm20_dg2,
            x.m22c, x_dg.dm22c_dg1,
            x.m22s, x_dg.dm22s_dg2
        };
    }

        template <typename T>
        inline py::array_t<FpfsShape> measure_fpfs_shape(
        double C0,
        const py::array_t<T> &x_array,
        const py::array_t<FpfsShapeletsResponse> &x_dg_array
    ) {
        auto x_r = x_array.template unchecked<1>();
        int nn = x_array.shape(0);
        py::array_t<FpfsShape> out(nn);
        auto out_r = out.mutable_unchecked<1>();
        auto xdg_r =  x_dg_array.unchecked<1>();
        for (ssize_t i = 0; i < nn; ++i) {
            out_r(i) = measure_fpfs_shape(
                C0,
                x_r(i),
                xdg_r(i)
            );
        }
        return out;
    };

        template <typename T>
        inline FpfsWeight measure_fpfs_wsel(
        double m00_min,
        double std_m00,
        double r2_min,
        double omega_r2,
        const T &x,
        const FpfsShapeletsResponse &x_dg
    ) {

        double omega_m00 = fpfs_cut_sigma_ratio * std_m00;
        // Selection on flux
        double w0l = math::ssfunc2(x.m00, m00_min, omega_m00);
        double dw0l = math::ssfunc2_deriv(x.m00, m00_min, omega_m00);
        double w0l_dg1 = dw0l * x_dg.dm00_dg1;
        double w0l_dg2 = dw0l * x_dg.dm00_dg2;

        double w0u = math::ssfunc2(-x.m00, -500, omega_m00);
        double dw0u = math::ssfunc2_deriv(-x.m00, -500, omega_m00);
        double w0u_dg1 = dw0u * -x_dg.dm00_dg1;
        double w0u_dg2 = dw0u * -x_dg.dm00_dg2;

        // Selection on size (lower limit)
        // (M00 + M20) / M00 > r2_min
        double r2l = x.m00 * (1.0 - r2_min) + x.m20;
        double w2l = math::ssfunc2(r2l, omega_r2, omega_r2);
        double dw2l = math::ssfunc2_deriv(r2l, omega_r2, omega_r2);
        double w2l_dg1 = dw2l * (
            x_dg.dm00_dg1 * (1.0 - r2_min) + x_dg.dm20_dg1
        );
        double w2l_dg2 = dw2l * (
            x_dg.dm00_dg2 * (1.0 - r2_min) + x_dg.dm20_dg2
        );

        // Compute the selection weights
        double wsel = w0l * w0u * w2l;
        double wsel_dg1 = (
            w0l_dg1 * w0u * w2l + w0l * w0u_dg1 * w2l + w0l * w0u * w2l_dg1
        );
        double wsel_dg2 = (
            w0l_dg2 * w0u * w2l + w0l * w0u_dg2 * w2l + w0l * w0u * w2l_dg2
        );

        return FpfsWeight{
            wsel, wsel_dg1, wsel_dg2
        };
    };

        inline FpfsWeight measure_fpfs_wdet0(
        double v_min,
        double omega_v,
        const FpfsDetect &x,
        const std::optional<FpfsDetect> &y=std::nullopt
    ) {

        FpfsDetect xx = y.has_value() ? x - *y * 2.0 : x;
        // det0 computation
        double det0 = math::ssfunc2(x.v0, v_min, omega_v);
        double det0_deriv = math::ssfunc2_deriv(x.v0, v_min, omega_v);
        double det0_dg1 = det0_deriv * (xx.dv0_dg1);
        double det0_dg2 = det0_deriv * (xx.dv0_dg2);

        // det1 computation
        double det1 = math::ssfunc2(x.v1, v_min, omega_v);
        double det1_deriv = math::ssfunc2_deriv(x.v1, v_min, omega_v);
        double det1_dg1 = det1_deriv * (xx.dv1_dg1);
        double det1_dg2 = det1_deriv * (xx.dv1_dg2);

        // det2 computation
        double det2 = math::ssfunc2(x.v2, v_min, omega_v);
        double det2_deriv = math::ssfunc2_deriv(x.v2, v_min, omega_v);
        double det2_dg1 = det2_deriv * (xx.dv2_dg1);
        double det2_dg2 = det2_deriv * (xx.dv2_dg2);

        // det3 computation
        double det3 = math::ssfunc2(x.v3, v_min, omega_v);
        double det3_deriv = math::ssfunc2_deriv(x.v3, v_min, omega_v);
        double det3_dg1 = det3_deriv * (xx.dv3_dg1);
        double det3_dg2 = det3_deriv * (xx.dv3_dg2);

        // Compute the weights
        double w = det0 * det1 * det2 * det3;
        double w_dg1 = (
            det0_dg1 * det1 * det2 * det3 + det0 * det1_dg1 * det2 * det3
        ) + (det0 * det1 * det2_dg1 * det3 + det0 * det1 * det2 * det3_dg1);
        double w_dg2 = (
            det0_dg2 * det1 * det2 * det3 + det0 * det1_dg2 * det2 * det3
        ) + (det0 * det1 * det2_dg2 * det3 + det0 * det1 * det2 * det3_dg2);


        return FpfsWeight{
            w, w_dg1, w_dg2
        };
    };

        inline py::array_t<FpfsWeight> measure_fpfs_wdet0(
        double v_min,
        double omega_v,
        const py::array_t<FpfsDetect> &x_array,
        const std::optional<py::array_t<FpfsDetect>> &y_array=std::nullopt
    ) {
        auto x_r = x_array.unchecked<1>();
        int nn = x_array.shape(0);
        py::array_t<FpfsWeight> out(nn);
        auto out_r = out.mutable_unchecked<1>();
        if (y_array.has_value()) {
            auto y_r = y_array->unchecked<1>();
            for (ssize_t i = 0; i < nn; ++i) {
                out_r(i) = measure_fpfs_wdet0(v_min, omega_v, x_r(i), y_r(i));
            }
        } else {
            for (ssize_t i = 0; i < nn; ++i) {
                out_r(i) = measure_fpfs_wdet0(v_min, omega_v, x_r(i));
            }
        }
        return out;
    };

        inline FpfsWeight measure_fpfs_wdet(
        double v_min,
        double omega_v,
        double pthres,
        const FpfsDetect &x,
        const std::optional<FpfsDetect> &y=std::nullopt
    ) {

        FpfsWeight w0 =  measure_fpfs_wdet0(v_min, omega_v, x, y);
        double wdet = math::ssfunc2(w0.w, pthres, fpfs_det_sigma2);
        double wdet_deriv = math::ssfunc2_deriv(w0.w, pthres, fpfs_det_sigma2);
        return FpfsWeight{
            wdet, wdet_deriv * w0.dw_dg1, wdet_deriv * w0.dw_dg2
        };
    };

        inline FpfsShape measure_fpfs(
        double C0,
        const FpfsShapelets &x,
        const std::optional<FpfsShapelets> &y=std::nullopt
    ){
        FpfsShapeletsResponse x_dg = measure_shapelets_dg(
            x, y
        );
        FpfsShape ell = measure_fpfs_shape(
            C0, x, x_dg
        );
        return ell;
    };

        inline FpfsCatalog measure_fpfs(
        double C0,
        double v_min,
        double omega_v,
        double pthres,
        double m00_min,
        double std_m00,
        double r2_min,
        double omega_r2,
        const FpfsDetect &x,
        const std::optional<FpfsDetect> &y=std::nullopt
    ){
        FpfsShapeletsResponse x_dg = measure_shapelets_dg(
            x, y
        );
        FpfsShape ell = measure_fpfs_shape(
            C0, x, x_dg
        );
        FpfsWeight wsel =  measure_fpfs_wsel(
            m00_min,
            std_m00,
            r2_min,
            omega_r2,
            x,
            x_dg
        );
        FpfsWeight wdet = measure_fpfs_wdet(
            v_min,
            omega_v,
            pthres,
            x,
            y
        );

        double w = wdet.w * wsel.w;
        double w_dg1 = wdet.w * wsel.dw_dg1 + wdet.dw_dg1 * wsel.w;
        double w_dg2 = wdet.w * wsel.dw_dg2 + wdet.dw_dg2 * wsel.w;
        return FpfsCatalog{
            ell.e1,
            ell.de1_dg1,
            ell.e2,
            ell.de2_dg2,
            ell.q1,
            ell.dq1_dg1,
            ell.q2,
            ell.dq2_dg2,
            w, w_dg1, w_dg2,
            x.m00, x_dg.dm00_dg1, x_dg.dm00_dg2,
            x.m20, x_dg.dm20_dg1, x_dg.dm20_dg2
        };
    };


        inline py::array_t<FpfsShape> measure_fpfs(
        double C0,
        const py::array_t<FpfsShapelets> &x_array,
        const std::optional<py::array_t<FpfsShapelets>> &y_array=std::nullopt
    ) {
        auto x_r = x_array.unchecked<1>();
        int nn = x_array.shape(0);
        py::array_t<FpfsShape> out(nn);
        auto out_r = out.mutable_unchecked<1>();
        if (y_array.has_value()) {
            auto y_r = y_array->unchecked<1>();
            for (ssize_t i = 0; i < nn; ++i) {
                out_r(i) = measure_fpfs(C0, x_r(i), y_r(i));
            }
        } else {
            for (ssize_t i = 0; i < nn; ++i) {
                out_r(i) = measure_fpfs(C0, x_r(i));
            }
        }
        return out;
    };


        inline py::array_t<FpfsCatalog> measure_fpfs(
        double C0,
        double v_min,
        double omega_v,
        double pthres,
        double m00_min,
        double std_m00,
        double r2_min,
        double omega_r2,
        const py::array_t<FpfsDetect> &x_array,
        const std::optional<py::array_t<FpfsDetect>> &y_array=std::nullopt
    ) {
        auto x_r = x_array.unchecked<1>();
        int nn = x_array.shape(0);
        py::array_t<FpfsCatalog> out(nn);
        auto out_r = out.mutable_unchecked<1>();
        if (y_array.has_value()) {
            auto y_r = y_array->unchecked<1>();
            for (ssize_t i = 0; i < nn; ++i) {
                out_r(i) = measure_fpfs(
                    C0,
                    v_min,
                    omega_v,
                    pthres,
                    m00_min,
                    std_m00,
                    r2_min,
                    omega_r2,
                    x_r(i),
                    y_r(i)
                );
            }
        } else {
            for (ssize_t i = 0; i < nn; ++i) {
                out_r(i) = measure_fpfs(
                    C0,
                    v_min,
                    omega_v,
                    pthres,
                    m00_min,
                    std_m00,
                    r2_min,
                    omega_r2,
                    x_r(i)
                );
            }
        }
        return out;
    };

        void pyExportFpfsCatalog(py::module_& fpfs);
    } // namespace fpfs
}

#endif // ANACAL_FPFS_CAT_H
