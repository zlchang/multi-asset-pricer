#include <gtest/gtest.h>
#include "pricer/Process.hpp"
#include "pricer/Payoff.hpp"
#include "pricer/MonteCarloEngine.hpp"
#include "pricer/RandomizedSobol.hpp"
#include "pricer/MathUtils.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>

// Simple linear regression slope on (x, y) vectors
double linearSlope(const std::vector<double>& x, const std::vector<double>& y) {
    const size_t n = x.size();
    double meanX = 0.0, meanY = 0.0;
    for (size_t i = 0; i < n; ++i) {
        meanX += x[i];
        meanY += y[i];
    }
    meanX /= static_cast<double>(n);
    meanY /= static_cast<double>(n);

    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dx = x[i] - meanX;
        num += dx * (y[i] - meanY);
        den += dx * dx;
    }
    return num / den;
}

TEST(SobolConvergence, FasterErrorDecayThanPRNGOnGeometricBasket) {
    pricer::MarketParams params;
    params.spots = Eigen::Vector3d(100.0, 105.0, 95.0);
    params.rates = Eigen::Vector3d(0.04, 0.04, 0.04);
    params.dividends = Eigen::Vector3d(0.015, 0.02, 0.01);
    params.vols = Eigen::Vector3d(0.18, 0.24, 0.22);
    params.corr = (Eigen::MatrixXd(3,3) << 1.0,  0.5,  0.3,
                   0.5,  1.0,  0.4,
                   0.3,  0.4,  1.0).finished();

    const double r = 0.04;
    const double strike = 100.;
    const double T = 1.0;
    const double exactPrice = pricer::MathUtils::analyticalGeometricBasketCall(
        params.spots, params.dividends, params.vols, params.corr, r, strike, T); // Replace with your analyticalGeometricBasketCall(...) result

    const std::vector<size_t> pathCounts = {
        1u << 8,   // 256
        1u << 10,  // 1,024
        1u << 12,  // 4,096
        1u << 14,  // 16,384
        1u << 16,  // 65,536
        1u << 18   // 262,144
    };

    std::vector<double> logN, sobolLogErr, prngLogErr;
    pricer::MultivariateGBM gbm(params);
    pricer::GeometricBasketCallPayoff payoff(strike);

    std::cout << "\n" << std::setw(10) << "N"
              << std::setw(16) << "Sobol Err"
              << std::setw(16) << "PRNG Err"
			  << std::setw(16) << "Sobol Price"
			  << std::setw(16) << "PRNG Price\n";
	size_t si = 0;
    for (size_t N : pathCounts) {
        // 1. PRNG baseline
        pricer::MonteCarloEngine mcEngine(gbm, N, 4, 43+si);
		const pricer::MCResult &prng = mcEngine.price(payoff, T, r);
        double prngPrice = prng.price;
		double prngStd = prng.stdError;
        double prngErr = std::max(std::abs(prngPrice - exactPrice), 1e-15);

        // 2. Sobol quasi-Monte Carlo (replace with your Sobol loop over i = 0..N-1)
        // Each point i uses 3 Sobol coordinates -> normInvCDF -> evolveStep(spots, T, Z)
		const pricer::MCResult &sobol = mcEngine.priceSobol(payoff, T, r);
        double sobolPrice = sobol.price; // Placeholder: plug in your Sobol accumulator here
        double sobolErr = std::max(std::abs(sobolPrice - exactPrice), 1e-15);
		double sobolStd = sobol.stdError;

        logN.push_back(std::log10(static_cast<double>(N)));
        sobolLogErr.push_back(std::log10(sobolErr));
        prngLogErr.push_back(std::log10(prngErr));
		//sobolLogErr.push_back(std::log10(sobolStd));
        //prngLogErr.push_back(std::log10(prngStd));


        std::cout << std::setw(10) << N
                  << std::setw(16) << sobolErr
                  << std::setw(16) << prngErr
				 << std::setw(16) << sobolPrice << " "<< sobolStd
				 <<std::setw(16) << prngPrice <<" "<< prngStd << "\n";
		si++;
    }

    double sobolSlope = linearSlope(logN, sobolLogErr);
    double prngSlope = linearSlope(logN, prngLogErr);

    std::cout << "Empirical log-log slopes -> Sobol: " << sobolSlope
              << ", PRNG: " << prngSlope << "\n";
	constexpr size_t dim = 3;
    constexpr size_t N = 1u << 16;
	pricer::SobolEngine sobol(dim, 43);
	std::vector<double> normalPoint(dim, 0.0);

    Eigen::Vector3d meanZ = Eigen::Vector3d::Zero();
    Eigen::Vector3d meanZ2 = Eigen::Vector3d::Zero();

    for (size_t n = 0; n < N; ++n) {
        for (size_t i = 0; i < dim; ++i) {
            sobol.nextNormal(normalPoint);
            const double z = normalPoint[i];
            meanZ[i] += z;
            meanZ2[i] += z * z;
        }
    }
    meanZ /= static_cast<double>(N);
	meanZ2 /= static_cast<double>(N);

    std::cout << "Sobol Z means: " << meanZ.transpose() << "\n"
              << "Sobol Z vars : " << (meanZ2.array() - meanZ.array().square()).transpose() << "\n";

    // Sobol should decay steeper (more negative slope) than PRNG's -0.5 theoretical rate
    EXPECT_LT(sobolSlope, -0.7);
}