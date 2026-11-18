#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <Eigen/Dense>
#include <stdexcept>

namespace pricer {

class MathUtils {
public:
    // Computes lower-triangular Cholesky factor L such that R = L * L^T
    static Eigen::MatrixXd computeCholesky(const Eigen::MatrixXd& corrMatrix) {
        if (corrMatrix.rows() != corrMatrix.cols()) {
            throw std::invalid_argument("Correlation matrix must be square.");
        }
        Eigen::LLT<Eigen::MatrixXd> llt(corrMatrix);
        if (llt.info() == Eigen::NumericalIssue) {
            throw std::runtime_error("Correlation matrix is not positive definite.");
        }
        return llt.matrixL();
    }

    // Standard normal cumulative distribution function (for analytical Margrabe benchmark)
    static double normCDF(double x) {
        return 0.5 * std::erfc(-x * M_SQRT1_2);
    }
};

} // namespace pricer

