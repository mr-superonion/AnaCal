#ifndef ANACAL_FPFS_CAT_H
#define ANACAL_FPFS_CAT_H


#include "base.h"

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
        double m00;
        double m00_g1;
        double m00_g2;
        double m20;
        double m20_g1;
        double m20_g2;
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
        double m00;
        double m00_g1;
        double m00_g2;
        double m20;
        double m20_g1;
        double m20_g2;
    };


    template <typename T>
    inline FpfsShapeletsResponse measure_shapelets_dg(
        const T& x,
        const std::optional<T>& y=std::nullopt
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
            e1, e1_g1, e2, e2_g2, q1, q1_g1, q2, q2_g2,
            x.m00, x_dg.m00_g1, x_dg.m00_g2,
            x.m20, x_dg.m20_g1, x_dg.m20_g2
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
        double std_r2,
        const T &x,
        const FpfsShapeletsResponse &x_dg
    ) {

        double sigma_m00 = fpfs_cut_sigma_ratio * std_m00;
        double sigma_r2 = fpfs_cut_sigma_ratio * std_r2;
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

    inline FpfsWeight measure_fpfs_wdet(
        double std_v,
        double pthres,
        const FpfsDetect &x,
        const std::optional<FpfsDetect> &y=std::nullopt
    ) {

        double sigma_v = fpfs_cut_sigma_ratio * std_v;
        double pcut = fpfs_pnr * std_v;
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
        double std_v,
        double pthres,
        double m00_min,
        double std_m00,
        double r2_min,
        double std_r2,
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
            std_r2,
            x,
            x_dg
        );
        FpfsWeight wdet = measure_fpfs_wdet(
            std_v,
            pthres,
            x,
            y
        );

        double w = wdet.w * wsel.w;
        double w_g1 = wdet.w * wsel.w_g1 + wdet.w_g1 * wsel.w;
        double w_g2 = wdet.w * wsel.w_g2 + wdet.w_g2 * wsel.w;
        return FpfsCatalog{
            ell.e1,
            ell.e1_g1,
            ell.e2,
            ell.e2_g2,
            ell.q1,
            ell.q1_g1,
            ell.q2,
            ell.q2_g2,
            w, w_g1, w_g2,
            x.m00, x_dg.m00_g1, x_dg.m00_g2,
            x.m20, x_dg.m20_g1, x_dg.m20_g2
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
        double std_v,
        double pthres,
        double m00_min,
        double std_m00,
        double r2_min,
        double std_r2,
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
                    std_v,
                    pthres,
                    m00_min,
                    std_m00,
                    r2_min,
                    std_r2,
                    x_r(i),
                    y_r(i)
                );
            }
        } else {
            for (ssize_t i = 0; i < nn; ++i) {
                out_r(i) = measure_fpfs(
                    C0,
                    std_v,
                    pthres,
                    m00_min,
                    std_m00,
                    r2_min,
                    std_r2,
                    x_r(i)
                );
            }
        }
        return out;
    };

    void pybindFpfsCatalog(py::module_& fpfs);
}

#endif // ANACAL_FPFS_CAT_H
