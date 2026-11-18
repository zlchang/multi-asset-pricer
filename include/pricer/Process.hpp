#pragma once
#include <Eigen/Dense>
#include <vector>

namespace pricer {

struct MarketParams {
    Eigen::VectorXd spots;      // S_0 for each asset
    Eigen::VectorXd rates;      // Risk-free rate r (scalar or per asset)
    Eigen::VectorXd dividends;  // Dividend yields q_i
    Eigen::VectorXd vols;       // Volatilities sigma_i
    Eigen::MatrixXd corr;       // Correlation matrix rho
};

class MultivariateGBM {
private:
    MarketParams params_;
    Eigen::MatrixXd choleskyL_;
    size_t numAssets_;

public:
    explicit MultivariateGBM(const MarketParams& params);

    size_t numAssets() const { return numAssets_; }
    const Eigen::VectorXd& initialSpots() const { return params_.spots; }

    // Evolves state S_t to S_{t+dt} given independent standard normal vector Z
    void evolveStep(Eigen::VectorXd& currentSpots, double dt, const Eigen::VectorXd& Z) const;
};

} // namespace pricer
