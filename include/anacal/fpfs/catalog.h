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
        double dm22c_dg2;
        double dm22s_dg1;
        double dm22s_dg2;
        double dm42c_dg1;
        double dm42c_dg2;
        double dm42s_dg1;
        double dm42s_dg2;
    };

    struct FpfsShape {
        double e1;
        double de1_dg1;
        double de1_dg2;
        double e2;
        double de2_dg1;
        double de2_dg2;
        double q1;
        double dq1_dg1;
        double dq1_dg2;
        double q2;
        double dq2_dg1;
        double dq2_dg2;
        double m00;
        double dm00_dg1;
        double dm00_dg2;
        double m20;
        double dm20_dg1;
        double dm20_dg2;
        double m22c;
        double dm22c_dg1;
        double dm22c_dg2;
        double m22s;
        double dm22s_dg1;
        double dm22s_dg2;
        double m42c;
        double dm42c_dg1;
        double dm42c_dg2;
        double m42s;
        double dm42s_dg1;
        double dm42s_dg2;
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
        double dm22c_dg2 = -std::sqrt(3.0) * xx.m44s;
        double dm22s_dg2 = (
            1.0 / std::sqrt(2.0)
        ) * (xx.m00 - xx.m40) + std::sqrt(3.0) * xx.m44c;
        double dm22s_dg1 = -std::sqrt(3.0) * xx.m44s;

        double dm42c_dg1 = (
            std::sqrt(6.0) / 2.0
        ) * (xx.m20 - xx.m60) - std::sqrt(5.0) * xx.m64c;
        double dm42c_dg2 = -std::sqrt(5.0) * xx.m64s;
        double dm42s_dg2 = (
            std::sqrt(6.0) / 2.0
        ) * (xx.m20 - xx.m60) + std::sqrt(5.0) * xx.m64c;
        double dm42s_dg1 = -std::sqrt(5.0) * xx.m64s;

        return FpfsShapeletsResponse{
            dm00_dg1, dm00_dg2, dm20_dg1,
            dm20_dg2, dm22c_dg1, dm22c_dg2,
            dm22s_dg1, dm22s_dg2, dm42c_dg1,
            dm42c_dg2, dm42s_dg1, dm42s_dg2
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
        double e1_dg2 = x_dg.dm22c_dg2 / denom
            - (x_dg.dm00_dg2 * x.m22c) / (denom * denom);

        // Compute ellipticity 2
        double e2 = x.m22s / denom;
        double e2_dg2 = x_dg.dm22s_dg2 / denom
            - (x_dg.dm00_dg2 * x.m22s) / (denom * denom);
        double e2_dg1 = x_dg.dm22s_dg1 / denom
            - (x_dg.dm00_dg1 * x.m22s) / (denom * denom);

        // Compute ellipticity 1 (4th order)
        double q1 = x.m42c / denom;
        double q1_dg1 = x_dg.dm42c_dg1 / denom
            - (x_dg.dm00_dg1 * x.m42c) / (denom * denom);
        double q1_dg2 = x_dg.dm42c_dg2 / denom
            - (x_dg.dm00_dg2 * x.m42c) / (denom * denom);

        // Compute ellipticity 2 (4th order)
        double q2 = x.m42s / denom;
        double q2_dg2 = x_dg.dm42s_dg2 / denom
            - (x_dg.dm00_dg2 * x.m42s) / (denom * denom);
        double q2_dg1 = x_dg.dm42s_dg1 / denom
            - (x_dg.dm00_dg1 * x.m42s) / (denom * denom);

        // Return the result as FpfsShape
        return FpfsShape{
            e1, e1_dg1, e1_dg2, e2, e2_dg1, e2_dg2,
            q1, q1_dg1, q1_dg2, q2, q2_dg1, q2_dg2,
            x.m00, x_dg.dm00_dg1, x_dg.dm00_dg2,
            x.m20, x_dg.dm20_dg1, x_dg.dm20_dg2,
            x.m22c, x_dg.dm22c_dg1, x_dg.dm22c_dg2,
            x.m22s, x_dg.dm22s_dg1, x_dg.dm22s_dg2,
            x.m42c, x_dg.dm42c_dg1, x_dg.dm42c_dg2,
            x.m42s, x_dg.dm42s_dg1, x_dg.dm42s_dg2
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


    void pyExportFpfsCatalog(py::module_& fpfs);
} // namespace fpfs
}

#endif // ANACAL_FPFS_CAT_H
