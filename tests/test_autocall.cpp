#include <gtest/gtest.h>
#include "pricer/Process.hpp"
#include "pricer/Payoff.hpp"
#include "pricer/MonteCarloEngine.hpp"

TEST(AutocallableTest, ZeroVolatilityDeterministicAutocall) {
    pricer::MarketParams params;
    params.spots = Eigen::Vector2d(100.0, 100.0);
    params.rates = Eigen::Vector2d(0.0, 0.0);
    params.dividends = Eigen::Vector2d(0.0, 0.0);
    params.vols = Eigen::Vector2d(0.0001, 0.0001); // Near zero vol -> deterministic drift

    params.corr = Eigen::Matrix2d::Identity();

    std::vector<double> obsTimes = {0.5, 1.0};
    // Autocall barrier at 90%, coupon 10% p.a.
    pricer::WorstOfAutocallablePayoff autocall(obsTimes, 0.90, 0.60, 0.10, 100.0);

    pricer::MultivariateGBM gbm(params);
    pricer::MonteCarloEngine engine(gbm, 1000000, 4, 42);

    auto res = engine.pricePathDependent(autocall, obsTimes, 0.0);

    // With zero vol and positive drift/spot, it should autocall at t=0.5 with 105 nominal returned
    EXPECT_NEAR(res.price, 105.0, 0.5);
}