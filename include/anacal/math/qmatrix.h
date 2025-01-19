#ifndef ANACAL_MATH_QMATRIX_H
#define ANACAL_MATH_QMATRIX_H

#include "../stdafx.h"
#include "qnumber.h"
#include <array>


namespace anacal {
namespace math {

// qmatrix structure with fixed dimensions
template <size_t Rows, size_t Cols>
struct qmatrix {
    std::array<qnumber, Cols * Rows> data{};

    // Access operator for matrix elements
    qnumber& operator()(size_t row, size_t col) {
        if (row >= Rows || col >= Cols) {
            throw std::out_of_range("qmatrix index out of bounds");
        }
        return data[row * Cols + col];
    };

    const qnumber& operator()(size_t row, size_t col) const {
        if (row >= Rows || col >= Cols) {
            throw std::out_of_range("qmatrix index out of bounds");
        }
        return data[row * Cols + col];
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

    // In-place addition
    qmatrix<Rows, Cols>& operator+=(const qmatrix<Rows, Cols>& other) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) += other(i, j);
            }
        }
        return *this;
    };

    // In-place subtraction
    qmatrix<Rows, Cols>& operator-=(const qmatrix<Rows, Cols>& other) {
        for (size_t i = 0; i < Rows; ++i) {
            for (size_t j = 0; j < Cols; ++j) {
                (*this)(i, j) -= other(i, j);
            }
        }
        return *this;
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
    template <size_t OtherCols>
    qmatrix<Rows, OtherCols> operator*(const qmatrix<Cols, OtherCols>& other) const {
        qmatrix<Rows, OtherCols> result{};
        for (size_t k = 0; k < Cols; ++k) {
            for (size_t i = 0; i < Rows; ++i) {
                for (size_t j = 0; j < OtherCols; ++j) {
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    };
};

} // end of math
} // end of anacal

#endif // ANACAL_MATH_QMATRIX_H
