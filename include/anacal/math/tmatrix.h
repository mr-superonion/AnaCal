#ifndef ANACAL_MATH_QMATRIX_H
#define ANACAL_MATH_QMATRIX_H

#include "../stdafx.h"
#include "tnumber.h"


namespace anacal {
namespace math {

// tmatrix structure with fixed dimensions
template <int Rows, int Cols>
struct tmatrix {
    int nelements = Cols * Rows;
    std::array<tnumber, Cols * Rows> data{};

    tmatrix() = default;

    tmatrix(const std::array<std::array<tnumber, Cols>, Rows>& data) {
        // Copy elements into the flat data array
        for (size_t i = 0; i < Rows; ++i) {
            size_t offset = i * Cols;
            for (size_t j = 0; j < Cols; ++j) {
                this->data[offset + j] = data[i][j];
            }
        }
    }

    tmatrix(const std::array<std::array<double, Cols>, Rows>& data) {
        // Copy elements into the flat data array
        for (size_t i = 0; i < Rows; ++i) {
            size_t offset = i * Cols;
            for (size_t j = 0; j < Cols; ++j) {
                this->data[offset + j].v = data[i][j];
            }
        }
    }

    tmatrix(py::array_t<double>& data) {
        // Copy elements into the flat data array
        auto data_r = data.unchecked<2>();
        for (size_t i = 0; i < Rows; ++i) {
            size_t offset = i * Cols;
            for (size_t j = 0; j < Cols; ++j) {
                this->data[offset + j].v = data_r(i, j);
            }
        }
    }

    // Access operator for matrix elements
    tnumber& operator()(size_t row, size_t col) {
        return data[row * Cols + col];
    };

    const tnumber& operator()(size_t row, size_t col) const {
        return data[row * Cols + col];
    };

    // Transpose the matrix
    tmatrix<Cols, Rows> transpose() const {
        tmatrix<Cols, Rows> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    };

    // tmatrix multiplication
    template <int OtherCols>
    tmatrix<Rows, OtherCols> operator*(const tmatrix<Cols, OtherCols>& other) const {
        tmatrix<Rows, OtherCols> result;
        tmatrix <OtherCols, Cols> other2 = other.transpose();
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < OtherCols; ++j) {
                for (size_t k = 0; k < Cols; ++k) {
                    result(i, j) = result(i, j) + (*this)(i, k) * other2(j, k);
                }
            }
        }
        return result;
    };

    // Addition of two matrices
    tmatrix<Rows, Cols> operator+(const tmatrix<Rows, Cols>& other) const {
        tmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
        return result;
    };

    // Subtraction of two matrices
    tmatrix<Rows, Cols> operator-(const tmatrix<Rows, Cols>& other) const {
        tmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) - other(i, j);
            }
        }
        return result;
    };

    // unary of this matrix
    tmatrix<Rows, Cols> operator-() const {
        tmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = -(*this)(i, j);
            }
        }
        return result;
    };

    // Scalar multiplication: tmatrix + tnumber
    tmatrix<Rows, Cols> operator+(tnumber scalar) const {
        tmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) + scalar;
            }
        }
        return result;
    };

    // Scalar multiplication: tnumber + tmatrix (friend function)
    friend tmatrix<Rows, Cols> operator+(
        tnumber scalar,
        const tmatrix<Rows, Cols>& matrix
    ) {
        return matrix + scalar;
    };

    // Scalar multiplication: tmatrix - tnumber
    tmatrix<Rows, Cols> operator-(tnumber scalar) const {
        tmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) - scalar;
            }
        }
        return result;
    };

    // Scalar multiplication: tnumber - tmatrix (friend function)
    friend tmatrix<Rows, Cols> operator-(
        tnumber scalar,
        const tmatrix<Rows, Cols>& matrix
    ) {
        tmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = scalar - matrix(i, j);
            }
        }
        return result;
    };

    // Scalar multiplication: tmatrix * tnumber
    tmatrix<Rows, Cols> operator*(tnumber scalar) const {
        tmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    };

    // Scalar multiplication: tnumber * tmatrix (friend function)
    friend tmatrix<Rows, Cols> operator*(
        tnumber scalar,
        const tmatrix<Rows, Cols>& matrix
    ) {
        return matrix * scalar;
    };

    // Scalar multiplication: tmatrix * double
    tmatrix<Rows, Cols> operator*(double scalar) const {
        tmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    };

    // Scalar multiplication: tnumber * double (friend function)
    friend tmatrix<Rows, Cols> operator*(
        double scalar,
        const tmatrix<Rows, Cols>& matrix
    ) {
        return matrix * scalar;
    };

    // Scalar multiplication: tmatrix / tnumber
    tmatrix<Rows, Cols> operator/(tnumber scalar) const {
        tmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) / scalar;
            }
        }
        return result;
    };

    // Scalar multiplication: tmatrix / double
    tmatrix<Rows, Cols> operator/(double scalar) const {
        tmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) / scalar;
            }
        }
        return result;
    };

    // to array
    py::array_t<double> to_array() const {
        auto result = py::array_t<double>({Cols, Rows, 5});
        auto res_r = result.mutable_unchecked<3>();
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                res_r(i, j, 0) = (*this)(i, j).v;
                res_r(i, j, 1) = (*this)(i, j).g1;
                res_r(i, j, 2) = (*this)(i, j).g2;
                res_r(i, j, 3) = (*this)(i, j).x1;
                res_r(i, j, 4) = (*this)(i, j).x2;
            }
        }
        return result;
    };
};

template <int N>
inline tmatrix<N, N> eye() {
    tmatrix<N, N> result;
    for (size_t i = 0; i < N; ++i) {
        result(i, i) = tnumber(1.0, 0.0, 0.0, 0.0, 0.0);
    }
    return result;
}; // indentity matrix

} // end of math
} // end of anacal

#endif // ANACAL_MATH_QMATRIX_H
