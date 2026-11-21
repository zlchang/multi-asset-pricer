#include <gtest/gtest.h>
#include "pricer/Process.hpp"
#include "pricer/Payoff.hpp"
#include "pricer/MonteCarloEngine.hpp"
#include "pricer/MathUtils.hpp"
#include <cmath>

TEST(MargrabeBenchmark, MatchesAnalyticalPriceWithin3StdErrors) {
    pricer::MarketParams params;
    params.spots = Eigen::Vector2d(100.0, 95.0);
    params.rates = Eigen::Vector2d(0.05, 0.05);
    params.dividends = Eigen::Vector2d(0.02, 0.03);
    params.vols = Eigen::Vector2d(0.20, 0.25);

    params.corr = (Eigen::Matrix2d() << 1.0, 0.6,
                   0.6, 1.0).finished();

    double T = 1.0;
    double exactPrice = pricer::MathUtils::analyticalMargrabe(100.0, 95.0, 0.02, 0.03, 0.20, 0.25, 0.6, T);

    pricer::MultivariateGBM gbm(params);
    pricer::MargrabeExchangePayoff payoff;
    //pricer::MonteCarloEngine engine(gbm, 250'000, 1, 42); // 1 step is exact for European GBM
	pricer::MonteCarloEngine engine(gbm, 500'000, 1, 42); // 1 step is exact for European GBM
    pricer::MCResult res = engine.price(payoff, T, 0.05);

    // Verify MC price falls within 3 standard errors of exact analytical price
    EXPECT_NEAR(res.price, exactPrice, 3.0 * res.stdError);
}
