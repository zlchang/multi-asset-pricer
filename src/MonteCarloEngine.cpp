// src/MonteCarloEngine.cpp
#include "pricer/MonteCarloEngine.hpp"
#include <future>
#include <random>
#include <cmath>

namespace pricer {

MCResult MonteCarloEngine::price(const Payoff& payoff, double maturity, double riskFreeRate) const {
    //std::mt19937_64 rng(seed_);
	size_t nproc = std::thread::hardware_concurrency();
	if(numPaths_ < nproc)
		nproc = numPaths_;
	
	size_t numPathProc = numPaths_ / nproc;
	size_t numPathRemn = numPaths_ % nproc;
	//printf("nproc=%zu, numPathProc=%zu\n", nproc, numPathProc);
	// Create a seed sequence from the master seed
    std::seed_seq seq{seed_};
	std::vector<unsigned int> thread_seeds(nproc);
    seq.generate(thread_seeds.begin(), thread_seeds.end());

    double sumPayoffs = 0.0;
    double sumSquaredPayoffs = 0.0;

	auto run = [&](unsigned int seed, size_t npaths)->std::pair<double, double>{
		std::mt19937_64 rng(seed);
		size_t nAssets = process_.numAssets();
		std::normal_distribution<double> normDist(0.0, 1.0);
		// Pre-allocate vectors outside the simulation loop to avoid heap fragmentation
		Eigen::VectorXd spots(nAssets);
		Eigen::VectorXd Z(nAssets);
	
		double dt = maturity / static_cast<double>(numSteps_);
		double discountFactor = std::exp(-riskFreeRate * maturity);
		double localSumPayoffs = 0.0;
		double localSumSquaredPayoffs = 0.0;

		for(size_t p = 0; p < npaths; ++p){
			spots = process_.initialSpots();
			for (size_t step = 0; step < numSteps_; ++step) {
				for (size_t i = 0; i < nAssets; ++i) {
					Z[i] = normDist(rng);
				}
				process_.evolveStep(spots, dt, Z);
			}	
			double discountedPayoff = discountFactor * payoff(spots);
			localSumPayoffs += discountedPayoff;
			localSumSquaredPayoffs += discountedPayoff * discountedPayoff;
		}
		return {localSumPayoffs, localSumSquaredPayoffs};
	};
	std::vector<std::future<std::pair<double, double>>> futures;
	futures.reserve(nproc);
    for (size_t p = 0; p < nproc; ++p) {
		size_t npaths = numPathProc + (p < numPathRemn ? 1:0);
		futures.push_back(std::async(std::launch::async, run, thread_seeds[p], npaths));
    }
	for(std::future<std::pair<double, double>> &fut: futures){
		const auto [lsum, lsumSquared] = fut.get();
		sumPayoffs += lsum;
		sumSquaredPayoffs += lsumSquared;
	}
    double mean = sumPayoffs / static_cast<double>(numPaths_);
    double variance = (sumSquaredPayoffs / static_cast<double>(numPaths_)) - (mean * mean);
    double stdError = std::sqrt(variance / static_cast<double>(numPaths_));

    return {mean, stdError};
}
//void MonteCarloEngine::run(unsigned int seed, size_t npaths, std::atomic<double> &sumPayoffs, std::atomic<double> &sumSquaredPayoffs, const Payoff& payoff, double maturity, double riskFreeRate) const
//{

//}
} // namespace pricer
