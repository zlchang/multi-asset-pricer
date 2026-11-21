#pragma once
#define _USE_MATH_DEFINES
#include <cmath>
#include <boost/math/special_functions/erf.hpp>
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
	// Inverse normal CDF (PPF) for converting uniform Sobol points to standard normals
    static double inverseNormCDF(double p) {
        //return M_SQRT2 * std::erfinv(2.0 * p - 1.0);
		return M_SQRT2 * boost::math::erf_inv(2.0 * p - 1.0);
    }
	
	static double analyticalMargrabe(double S1, double S2, double q1, double q2,
                          double sigma1, double sigma2, double rho, double T) {
    double sigma = std::sqrt(sigma1 * sigma1 + sigma2 * sigma2 - 2.0 * rho * sigma1 * sigma2);
    double d1 = (std::log(S1 / S2) + (q2 - q1 + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    return S1 * std::exp(-q1 * T) * pricer::MathUtils::normCDF(d1)
         - S2 * std::exp(-q2 * T) * pricer::MathUtils::normCDF(d2);
}

	static double analyticalGeometricBasketCall(const Eigen::Vector3d& spots,
									const Eigen::Vector3d& dividends,
                                     const Eigen::Vector3d& vols,
                                     const Eigen::Matrix3d& corr,
                                     double r,
                                     double strike,
                                     double T) {
    const double g0 = std::cbrt(spots[0] * spots[1] * spots[2]);

    double muG = 0.0;
    for (int i = 0; i < 3; ++i) {
        muG += (r - dividends[i] - 0.5 * vols[i] * vols[i]);
    }
    muG /= 3.0;

    double varG = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            varG += corr(i, j) * vols[i] * vols[j];
        }
    }
    varG /= 9.0;

    const double sigmaG = std::sqrt(varG);
    const double qG = r - muG - 0.5 * sigmaG * sigmaG;

    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(g0 / strike) + (r - qG + 0.5 * sigmaG * sigmaG) * T) / (sigmaG * sqrtT);
    const double d2 = d1 - sigmaG * sqrtT;

    return g0 * std::exp(-qG * T) * pricer::MathUtils::normCDF(d1)
         - strike * std::exp(-r * T) * pricer::MathUtils::normCDF(d2);
}
};
} // namespace pricer

