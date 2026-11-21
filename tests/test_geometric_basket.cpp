#include <gtest/gtest.h>
#include "pricer/Process.hpp"
#include "pricer/Payoff.hpp"
#include "pricer/MonteCarloEngine.hpp"
#include "pricer/MathUtils.hpp"
#include <cmath>

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

    const double exactPrice = pricer::MathUtils::analyticalGeometricBasketCall(
        params.spots, params.dividends, params.vols, params.corr, r, strike, T);

    pricer::MultivariateGBM gbm(params);
    pricer::GeometricBasketCallPayoff payoff(strike);
    pricer::MonteCarloEngine engine(gbm,  1u<<18, 4, 42);

    const pricer::MCResult res = engine.priceSobol(payoff, T, r);
	std::cout << std::setw(16) << res.price
                  << std::setw(16) << res.stdError
                  << std::setw(16) << exactPrice << "\n";

    EXPECT_NEAR(res.price, exactPrice, 3.0 * res.stdError);
}