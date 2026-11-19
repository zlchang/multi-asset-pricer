// include/pricer/MonteCarloEngine.hpp
#pragma once
#include "Process.hpp"
#include "Payoff.hpp"
#include <cstddef>
#include <atomic>
#include <thread>

namespace pricer {

struct MCResult {
    double price;
    double stdError;
};

class MonteCarloEngine {
private:
    const MultivariateGBM& process_;
    size_t numPaths_;
    size_t numSteps_;
    unsigned int seed_;

public:
    MonteCarloEngine(const MultivariateGBM& process, size_t numPaths, size_t numSteps, unsigned int seed = 42)
        : process_(process), numPaths_(numPaths), numSteps_(numSteps), seed_(seed) {}

    MCResult price(const Payoff& payoff, double maturity, double riskFreeRate) const;
	MCResult pricePathDependent(const Payoff& payoff, const std::vector<double>& obsTimes, double riskFreeRate) const;
};

} // namespace pricer
