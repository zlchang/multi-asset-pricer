// src/Process.cpp
#include "pricer/Process.hpp"
#include "pricer/MathUtils.hpp"
#include <cmath>

namespace pricer {

MultivariateGBM::MultivariateGBM(const MarketParams& params)
    : params_(params), numAssets_(params.spots.size()) {
    choleskyL_ = MathUtils::computeCholesky(params_.corr);
}

void MultivariateGBM::evolveStep(Eigen::VectorXd& currentSpots, double dt, const Eigen::VectorXd& Z) const {
    Eigen::VectorXd correlatedZ = choleskyL_ * Z;
    double sqrtDt = std::sqrt(dt);

    for (size_t i = 0; i < numAssets_; ++i) {
        double r = params_.rates[i];
        double q = params_.dividends[i];
        double sigma = params_.vols[i];

        double drift = (r - q - 0.5 * sigma * sigma) * dt;
        double diffusion = sigma * sqrtDt * correlatedZ[i];
        currentSpots[i] *= std::exp(drift + diffusion);
    }
}

} // namespace pricer
