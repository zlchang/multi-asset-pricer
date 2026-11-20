#include <gtest/gtest.h>
#include "pricer/Process.hpp"
#include "pricer/Payoff.hpp"
#include "pricer/MonteCarloEngine.hpp"
#include "pricer/MathUtils.hpp"
#include <cmath>

double analyticalMargrabe(double S1, double S2, double q1, double q2,
                          double sigma1, double sigma2, double rho, double T) {
    double sigma = std::sqrt(sigma1 * sigma1 + sigma2 * sigma2 - 2.0 * rho * sigma1 * sigma2);
    double d1 = (std::log(S1 / S2) + (q2 - q1 + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    return S1 * std::exp(-q1 * T) * pricer::MathUtils::normCDF(d1)
         - S2 * std::exp(-q2 * T) * pricer::MathUtils::normCDF(d2);
}

TEST(MargrabeBenchmark, MatchesAnalyticalPriceWithin3StdErrors) {
    pricer::MarketParams params;
    params.spots = Eigen::Vector2d(100.0, 95.0);
    params.rates = Eigen::Vector2d(0.05, 0.05);
    params.dividends = Eigen::Vector2d(0.02, 0.03);
    params.vols = Eigen::Vector2d(0.20, 0.25);

    params.corr = (Eigen::Matrix2d() << 1.0, 0.6,
                   0.6, 1.0).finished();

    double T = 1.0;
    double exactPrice = analyticalMargrabe(100.0, 95.0, 0.02, 0.03, 0.20, 0.25, 0.6, T);

    pricer::MultivariateGBM gbm(params);
    pricer::MargrabeExchangePayoff payoff;
    //pricer::MonteCarloEngine engine(gbm, 250'000, 1, 42); // 1 step is exact for European GBM
	pricer::MonteCarloEngine engine(gbm, 500'000, 1, 42); // 1 step is exact for European GBM
    pricer::MCResult res = engine.price(payoff, T, 0.05);

    // Verify MC price falls within 3 standard errors of exact analytical price
    EXPECT_NEAR(res.price, exactPrice, 3.0 * res.stdError);
}
