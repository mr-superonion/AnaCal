#ifndef ANACAL_MATH_QMATRIX_H
#define ANACAL_MATH_QMATRIX_H

#include "../stdafx.h"
#include "qnumber.h"
#include <array>


namespace anacal {
namespace math {

// qmatrix structure with fixed dimensions
template <int Rows, int Cols>
struct qmatrix {
    int nelements = Cols * Rows;
    std::array<qnumber, Cols * Rows> data{};

    qmatrix() = default;

    qmatrix(const std::array<std::array<qnumber, Cols>, Rows>& data) {
        // Copy elements into the flat data array
        for (size_t i = 0; i < Rows; ++i) {
            size_t offset = i * Cols;
            for (size_t j = 0; j < Cols; ++j) {
                this->data[offset + j] = data[i][j];
            }
        }
    }

    qmatrix(const std::array<std::array<double, Cols>, Rows>& data) {
        // Copy elements into the flat data array
        for (size_t i = 0; i < Rows; ++i) {
            size_t offset = i * Cols;
            for (size_t j = 0; j < Cols; ++j) {
                this->data[offset + j].v = data[i][j];
            }
        }
    }

    qmatrix(py::array_t<double>& data) {
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
    qnumber& operator()(size_t row, size_t col) {
        return data[row * Cols + col];
    };

    const qnumber& operator()(size_t row, size_t col) const {
        return data[row * Cols + col];
    };

    // Transpose the matrix
    qmatrix<Cols, Rows> transpose() const {
        qmatrix<Cols, Rows> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    };

    // qmatrix multiplication
    template <int OtherCols>
    qmatrix<Rows, OtherCols> operator*(const qmatrix<Cols, OtherCols>& other) const {
        qmatrix<Rows, OtherCols> result;
        qmatrix <OtherCols, Cols> other2 = other.transpose();
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
    qmatrix<Rows, Cols> operator+(const qmatrix<Rows, Cols>& other) const {
        qmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) + other(i, j);
            }
        }
        return result;
    };

    // Subtraction of two matrices
    qmatrix<Rows, Cols> operator-(const qmatrix<Rows, Cols>& other) const {
        qmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) - other(i, j);
            }
        }
        return result;
    };

    // unary of this matrix
    qmatrix<Rows, Cols> operator-() const {
        qmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = -(*this)(i, j);
            }
        }
        return result;
    };

    // Scalar multiplication: qmatrix + qnumber
    qmatrix<Rows, Cols> operator+(qnumber scalar) const {
        qmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) + scalar;
            }
        }
        return result;
    };

    // Scalar multiplication: qnumber + qmatrix (friend function)
    friend qmatrix<Rows, Cols> operator+(
        qnumber scalar,
        const qmatrix<Rows, Cols>& matrix
    ) {
        return matrix + scalar;
    };

    // Scalar multiplication: qmatrix - qnumber
    qmatrix<Rows, Cols> operator-(qnumber scalar) const {
        qmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) - scalar;
            }
        }
        return result;
    };

    // Scalar multiplication: qnumber - qmatrix (friend function)
    friend qmatrix<Rows, Cols> operator-(
        qnumber scalar,
        const qmatrix<Rows, Cols>& matrix
    ) {
        qmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = scalar - matrix(i, j);
            }
        }
        return result;
    };

    // Scalar multiplication: qmatrix * qnumber
    qmatrix<Rows, Cols> operator*(qnumber scalar) const {
        qmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    };

    // Scalar multiplication: qnumber * qmatrix (friend function)
    friend qmatrix<Rows, Cols> operator*(
        qnumber scalar,
        const qmatrix<Rows, Cols>& matrix
    ) {
        return matrix * scalar;
    };

    // Scalar multiplication: qmatrix * double
    qmatrix<Rows, Cols> operator*(double scalar) const {
        qmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) * scalar;
            }
        }
        return result;
    };

    // Scalar multiplication: qnumber * double (friend function)
    friend qmatrix<Rows, Cols> operator*(
        double scalar,
        const qmatrix<Rows, Cols>& matrix
    ) {
        return matrix * scalar;
    };

    // Scalar multiplication: qmatrix / qnumber
    qmatrix<Rows, Cols> operator/(qnumber scalar) const {
        qmatrix<Rows, Cols> result;
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                result(i, j) = (*this)(i, j) / scalar;
            }
        }
        return result;
    };

    // Scalar multiplication: qmatrix / double
    qmatrix<Rows, Cols> operator/(double scalar) const {
        qmatrix<Rows, Cols> result;
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
inline qmatrix<N, N> eye() {
    qmatrix<N, N> result;
    for (size_t i = 0; i < N; ++i) {
        result(i, i) = qnumber(1.0, 0.0, 0.0, 0.0, 0.0);
    }
    return result;
}; // indentity matrix

} // end of math
} // end of anacal

#endif // ANACAL_MATH_QMATRIX_H
