// src/MonteCarloEngine.cpp
#include "pricer/MonteCarloEngine.hpp"
#include "pricer/RandomizedSobol.hpp"
#include <future>
#include <random>
#include <cmath>

namespace pricer {

MCResult MonteCarloEngine::price(const Payoff& payoff, double maturity, double riskFreeRate) const {
    //std::mt19937_64 rng(seed_);
	size_t nproc = std::thread::hardware_concurrency();
	if(numPaths_ < nproc)
		nproc = numPaths_;
	if(nproc == 0) nproc = 2;
	
	const size_t numPathProc = numPaths_ / nproc;
	const size_t numPathRemn = numPaths_ % nproc;
	//printf("nproc=%zu, numPathProc=%zu\n", nproc, numPathProc);
	// Create a seed sequence from the master seed
    //std::seed_seq seq{seed_};
	//std::vector<unsigned int> thread_seeds(nproc);
    //seq.generate(thread_seeds.begin(), thread_seeds.end());

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
		return std::pair<double, double>{localSumPayoffs, localSumSquaredPayoffs};
	};
	std::vector<std::future<std::pair<double, double>>> futures;
	futures.reserve(nproc);
    for (size_t p = 0; p < nproc; ++p) {
		size_t npaths = numPathProc + (p < numPathRemn ? 1:0);
		futures.push_back(std::async(std::launch::async, run, seed_+p, npaths));
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

MCResult MonteCarloEngine::pricePathDependent(const Payoff& payoff, 
                                              const std::vector<double>& obsTimes, 
                                              double riskFreeRate) const {
	size_t nproc = std::thread::hardware_concurrency();
	if(numPaths_ < nproc)
		nproc = numPaths_;
	if(nproc == 0) nproc = 2;
	
	const size_t numPathProc = numPaths_ / nproc;
	const size_t numPathRemn = numPaths_ % nproc;
	//printf("nproc=%zu, numPathProc=%zu\n", nproc, numPathProc);
	// Create a seed sequence from the master seed
    //std::seed_seq seq{seed_};
	//std::vector<unsigned int> thread_seeds(nproc);
    //seq.generate(thread_seeds.begin(), thread_seeds.end());

    const size_t nAssets = process_.numAssets();
    const size_t numObs = obsTimes.size();
    const double discountFactor = std::exp(-riskFreeRate * obsTimes.back());

    double sumPayoffs = 0.0;
    double sumSquaredPayoffs = 0.0;
    //size_t totalPathsSimulated = numPaths_ * 2; // 2x due to antithetic pairing

	auto run = [&](unsigned int seed, int npaths)->std::pair<double, double>{ 
		// Allocate storage for paths: path[0] = S_0, path[1] = S(t_1), etc.
		PathMatrix path1(numObs + 1);
		PathMatrix path2(numObs + 1); // For antithetic variates
		for (auto& p : path1) p.resize(nAssets);
		for (auto& p : path2) p.resize(nAssets);

		Eigen::VectorXd Z(nAssets);
		std::mt19937_64 rng(seed);
		double lSumPayoffs = 0.0;
		double lSumSquaredPayoffs = 0.0;
		std::normal_distribution<double> normDist(0.0, 1.0);
		for (size_t p = 0; p < npaths; ++p) {
			// Initialize spots
			path1[0] = process_.initialSpots();
			path2[0] = process_.initialSpots();

			double prevT = 0.0;
			for (size_t step = 0; step < numObs; ++step) {
				double t = obsTimes[step];
				double dt = t - prevT;

				// Generate standard normal increments
				for (size_t i = 0; i < nAssets; ++i) {
					Z[i] = normDist(rng);
				}

				// Evolve path 1 with +Z, path 2 with -Z (Antithetic pair)
				path1[step + 1] = path1[step];
				path2[step + 1] = path2[step];

				process_.evolveStep(path1[step + 1], dt, Z);
				process_.evolveStep(path2[step + 1], dt, -Z);

				prevT = t;
			}

			double pay1 = payoff.evaluate(path1, process_.initialSpots());
			double pay2 = payoff.evaluate(path2, process_.initialSpots());

			double avgPayoff = 0.5 * (pay1 + pay2);
			double discountedPayoff = discountFactor * avgPayoff;
			
			lSumPayoffs += discountedPayoff;
			lSumSquaredPayoffs += discountedPayoff * discountedPayoff;
		}
		return std::pair<double, double>{lSumPayoffs, lSumSquaredPayoffs};	
    };
	std::vector<std::future<std::pair<double, double>>> futures;
	futures.reserve(nproc);
	for(size_t i = 0; i < nproc; i++){
		size_t nps = numPathProc + (i < numPathRemn ? 1:0);
		futures.push_back(std::async(std::launch::async, run, seed_+i, nps));
	}
	for(auto &fut: futures){
		auto [lSum, lSumSquared] = fut.get();
		sumPayoffs += lSum;
		sumSquaredPayoffs += lSumSquared;
	}
    double mean = sumPayoffs / static_cast<double>(numPaths_);
    double variance = (sumSquaredPayoffs / static_cast<double>(numPaths_)) - (mean * mean);
    double stdError = std::sqrt(variance / static_cast<double>(numPaths_));

    return {mean, stdError};
}

MCResult MonteCarloEngine::priceSobol(const Payoff& payoff, 
									double maturity,
									double riskFreeRate) const {
	size_t nproc = std::thread::hardware_concurrency();
	if(numPaths_ < nproc)
		nproc = numPaths_;
	if(nproc == 0) nproc = 2;
	//nproc = 1;
	const size_t numPathProc = numPaths_ / nproc;
	const size_t numPathRemn = numPaths_ % nproc;
	//printf("nproc=%zu, numPathProc=%zu\n", nproc, numPathProc);
	// Create a seed sequence from the master seed
    //std::seed_seq seq{seed_};
	//std::vector<unsigned int> thread_seeds(nproc);
    //seq.generate(thread_seeds.begin(), thread_seeds.end());

    const size_t nAssets = process_.numAssets();
    //size_t numObs = obsTimes.size();
	const size_t numObs = numSteps_;
    //double discountFactor = std::exp(-riskFreeRate * obsTimes.back());
	double discountFactor = std::exp(-riskFreeRate * maturity);

    // Allocate storage for paths: path[0] = S_0, path[1] = S(t_1), etc.

    double sumPayoffs = 0.0;
    double sumSquaredPayoffs = 0.0;
    //size_t totalPathsSimulated = numPaths_ * 2; // 2x due to antithetic pairing
	size_t dim = nAssets;

	auto run = [&](unsigned int seed, size_t npaths)->std::pair<double, double>{
		//PathMatrix path1(numObs + 1);
		//PathMatrix path2(numObs + 1);
		//for (auto& p : path2) p.resize(nAssets);
		//for (auto& p : path1) p.resize(nAssets);
		Eigen::VectorXd spot1(nAssets);
		Eigen::VectorXd spot2(nAssets);
		
		Eigen::VectorXd Z(nAssets);
		//printf("dim=%zu seed=%ud\n", dim, seed);
		//std::mt19937_64 rng(seed);
		//std::uniform_real_distribution<double> dist(0.0, 1.0);
		//std::vector<double> shift(dim);
		//std::vector<double> uniformSobol(dim, 0.0);
		//std::vector<double> uniformPoint(dim, 0.0);
		SobolEngine sobol(dim, seed);
		std::vector<double> normalPoint(dim, 0.0);
		
		double lSumPayoffs = 0.0;
		double lSumSquaredPayoffs = 0.0;
		const int nreps = 1u << 8;
		const int nri = npaths/nreps;
		const int nrr = npaths%nreps;
		std::vector<size_t> rpaths(nreps);
		for(size_t i = 0; i < nreps; i++){
			rpaths[i] = nri + (i < nrr ? 1 : 0);
		}
		//std::normal_distribution<double> normDist(0.0, 1.0);
		for(size_t rp = 0; rp < nreps; ++rp){
			sobol.redraw();
		for (size_t p = 0; p < rpaths[rp]; ++p) {
			// Initialize spots
			//for (std::size_t j = 0; j < dim; ++j) {
				//shift[j] = dist(rng);
			//}
			//sobol.nextUniform(uniformSobol);
			//applyDigitalShift(uniformSobol, shift, uniformPoint);
			//if(p == 0) printf("normalPoint=%lf seed=%ud\n", normalPoint[0], seed);
			//double prevT = 0.0;
			double dt = maturity / static_cast<double>(numSteps_);
			spot1 = process_.initialSpots();
			//spot2 = process_.initialSpots();
			for (size_t step = 0; step < numObs; ++step) {
				//double t = obsTimes[step];
				//double dt = t - prevT;
				size_t offset = step * nAssets;
				// Generate standard normal increments
				for (size_t i = 0; i < nAssets; ++i) {
					sobol.nextNormal(normalPoint);
					//Z[i] = MathUtils::inverseNormCDF(uniformPoint[offset + i]);
					Z[i] = normalPoint[offset+i];
				}
				//if(p < 5){
					//printf("%lf %lf %lf\n", Z[0], Z[1], Z[2]);
				//}

				//path1[step + 1] = path1[step];
				//path2[step + 1] = path2[step];
				
				process_.evolveStep(spot1, dt, Z);
				//process_.evolveStep(spot2, dt, -Z);

				//prevT = t;
			}

			//double pay = payoff.evaluate(path[numObs], process_.initialSpots());
			double pay1 = payoff(spot1);
			//double pay2 = payoff(spot2);

			//double avgPayoff = 0.5 * (pay1 + pay2);
			double avgPayoff = pay1;
			double discountedPayoff = discountFactor * avgPayoff;
			lSumPayoffs += discountedPayoff;
			lSumSquaredPayoffs += discountedPayoff * discountedPayoff;
		}
		}
		return std::pair<double, double>{lSumPayoffs, lSumSquaredPayoffs};	
    };
	std::vector<std::future<std::pair<double, double>>> futures;
	futures.reserve(nproc);
	for(size_t i = 0; i < nproc; i++){
		size_t nps = numPathProc + (i < numPathRemn ? 1:0);
		futures.push_back(std::async(std::launch::async, run, seed_+i, nps));
	}
	for(auto &fut: futures){
		auto [lSum, lSumSquared] = fut.get();
		sumPayoffs += lSum;
		sumSquaredPayoffs += lSumSquared;
	}
    double mean = sumPayoffs / static_cast<double>(numPaths_);
    double variance = (sumSquaredPayoffs / static_cast<double>(numPaths_)) - (mean * mean);
    double stdError = std::sqrt(variance / static_cast<double>(numPaths_));

    return {mean, stdError};
}

} // namespace pricer
