#ifndef ANACAL_MATH_QN_H
#define ANACAL_MATH_QN_H

#include <cmath>
#include <ostream>

// Class for Quintuple number

namespace anacal {
namespace math {


// Update B_inv according to the formula
template <size_t N>
void update_bfgs_BInverse(
        const std::array<qnumber, N * N>& H,
        const std::array<>& s,
        const std::array<double>& y,
        std::vector<std::vector<double>>& B_inv_new
) {
    size_t n = s.size();
    double sTy = dotProduct(s, y);

    if (sTy == 0.0) {
        throw std::runtime_error("s^T * y is zero, cannot update B_inverse.");
    }

    // Compute y^T * B^-1 * y
    std::vector<double> B_inv_y(n);
    matrixVectorMultiply(B_inv, y, B_inv_y);
    double yTB_inv_y = dotProduct(y, B_inv_y);

    // Compute the first term: B^-1
    B_inv_new = B_inv;

    // Compute the second term: (s^T * y + y^T * B^-1 * y) * (s * s^T) / (s^T * y)^2
    std::vector<std::vector<double>> s_outer_s(n, std::vector<double>(n));
    outerProduct(s, s, s_outer_s);
    double coeff1 = (sTy + yTB_inv_y) / (sTy * sTy);
    scalarMultiply(s_outer_s, coeff1);

    matrixAdd(B_inv_new, s_outer_s, B_inv_new);

    // Compute the third term: (B^-1 * y * s^T + s * y^T * B^-1) / (s^T * y)
    std::vector<std::vector<double>> term3(n, std::vector<double>(n));
    std::vector<std::vector<double>> B_inv_y_outer_s(n, std::vector<double>(n));
    outerProduct(B_inv_y, s, B_inv_y_outer_s);
    scalarMultiply(B_inv_y_outer_s, 1.0 / sTy);

    std::vector<std::vector<double>> s_outer_B_inv_y(n, std::vector<double>(n));
    outerProduct(s, B_inv_y, s_outer_B_inv_y);
    scalarMultiply(s_outer_B_inv_y, 1.0 / sTy);

    matrixAdd(B_inv_y_outer_s, s_outer_B_inv_y, term3);

    matrixSubtract(B_inv_new, term3, B_inv_new);
}

} // namespace math
} // namespace anacal

#endif // ANACAL_MATH_QN_H
