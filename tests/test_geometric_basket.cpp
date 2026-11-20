#include <gtest/gtest.h>
#include "pricer/Process.hpp"
#include "pricer/Payoff.hpp"
#include "pricer/MonteCarloEngine.hpp"
#include "pricer/MathUtils.hpp"
#include <cmath>

double analyticalGeometricBasketCall(const Eigen::Vector3d& spots,
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

TEST(GeometricBasketBenchmark, ThreeAssetsMatchAnalyticalWithin3StdErrors) {
    pricer::MarketParams params;
    params.spots = Eigen::Vector3d(100.0, 105.0, 95.0);
    params.rates = Eigen::Vector3d(0.04, 0.04, 0.04);
    params.dividends = Eigen::Vector3d(0.015, 0.02, 0.01);
    params.vols = Eigen::Vector3d(0.18, 0.24, 0.22);

    params.corr = (Eigen::MatrixXd(3,3) << 1.0,  0.5,  0.3,
                   0.5,  1.0,  0.4,
                   0.3,  0.4,  1.0).finished();

    const double r = 0.04;
    const double strike = 100.0;
    const double T = 1.0;

    const double exactPrice = analyticalGeometricBasketCall(
        params.spots, params.dividends, params.vols, params.corr, r, strike, T);

    pricer::MultivariateGBM gbm(params);
    pricer::GeometricBasketCallPayoff payoff(strike);
    pricer::MonteCarloEngine engine(gbm, 300'000, 1, 42);

    const pricer::MCResult res = engine.price(payoff, T, r);

    EXPECT_NEAR(res.price, exactPrice, 3.0 * res.stdError);
}