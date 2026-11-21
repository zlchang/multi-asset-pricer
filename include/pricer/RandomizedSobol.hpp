#pragma once
#include <vector>
#include <random>
#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include "pricer/MathUtils.hpp"

namespace pricer {


//Boost.Random's sobol engine.
class SobolEngine {
public:
    explicit SobolEngine(size_t dim) : dim_(dim), engine_(dim){
    }
	explicit SobolEngine(size_t dim, uint64_t seed) : dim_(dim), engine_(dim), rng_(seed), dist_(0., 1.0), 
	shiftU_(dim, 0.), scratchU_(dim, 0.){
		for (size_t j = 0; j < dim_; ++j) {
			shiftU_[j] = dist_(rng_);
			//printf("shift=%lf\n", shiftU_[j]);
		}
    }
	size_t dimension() const noexcept { return dim_; }

    void reset() {
        engine_.seed();
    }
	void redraw(){
		for (size_t j = 0; j < dim_; ++j) {
			shiftU_[j] = dist_(rng_);
			//printf("shift=%lf\n", shiftU_[j]);
		}
		reset();
	}

    // Writes dim_ uniform values in [0, 1) into out
    void nextUniform(std::vector<double>& out) {
		if (out.size() != dim_) {
            out.resize(dim_);
        }

		boost::random::uniform_01<double> dist;
		// Generates a Cranley-Patterson shifted Sobol point in [0, 1)^dim
        for (size_t j = 0; j < dim_; ++j) {
            double u = dist(engine_)+shiftU_[j];
			if (u >= 1.0) {
				u -= 1.0;
			}
            // Keep strictly inside (0, 1) so inverseNormCDF never hits +-inf
            out[j] = std::clamp(u, 1e-12, 1.0 - 1e-12);
        }
    }
	void nextNormal(std::vector<double>& out) {
		nextUniform(scratchU_);
        if (out.size() != dim_) {
            out.resize(dim_);
        }
		
        for (size_t j = 0; j < dim_; ++j) {
            out[j] = MathUtils::inverseNormCDF(scratchU_[j]);
        }
	}

	//void applyDigitalShift(const std::vector<double>& sobolU, const std::vector<double>& shift, std::vector<double>& outU) {



private:
    size_t dim_;
	boost::random::sobol engine_;
	std::mt19937_64 rng_;
	std::vector<double> scratchU_;
	//std::vector<double> sobolU_;
	std::vector<double> shiftU_;
	std::uniform_real_distribution<double> dist_;

};

} // namespace pricer