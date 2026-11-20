// include/pricer/Payoff.hpp
#pragma once
#include <Eigen/Dense>
#include <algorithm>

namespace pricer {

// Represents a path of asset prices across discrete observation times:
// dimensions: [numSteps + 1][numAssets] (or stored linearly for cache locality)
using PathMatrix = std::vector<Eigen::VectorXd>;

class Payoff {
public:
    virtual ~Payoff() = default;
    // Evaluates payoff given asset prices at maturity S_T
    virtual double operator()(const Eigen::VectorXd& terminalSpots) const = 0;
	virtual double evaluate(const PathMatrix& path, const Eigen::VectorXd& initialSpots) const = 0;
};

// Exchange Option: max(S_1(T) - S_2(T), 0)
class MargrabeExchangePayoff : public Payoff {
public:
    double operator()(const Eigen::VectorXd& terminalSpots) const override {
        return std::max(terminalSpots[0] - terminalSpots[1], 0.0);
    }
	double evaluate(const PathMatrix& path, const Eigen::VectorXd& initialSpots) const override { return 0.;};
};

class EuropeanCallPayoff : public Payoff {
private:
    double strike_;
public:
    explicit EuropeanCallPayoff(double strike) : strike_(strike) {}
    
    double operator()(const Eigen::VectorXd& terminalSpots) const override {
        return std::max(terminalSpots[0] - strike_, 0.0);
    }
	double evaluate(const PathMatrix& path, const Eigen::VectorXd& initialSpots) const override { return 0.;};
};

class EuropeanPutPayoff : public Payoff {
private:
    double strike_;
public:
    explicit EuropeanPutPayoff(double strike) : strike_(strike) {}
    
    double operator()(const Eigen::VectorXd& terminalSpots) const override {
        return std::max(strike_ - terminalSpots[0], 0.0);
    }
	double evaluate(const PathMatrix& path, const Eigen::VectorXd& initialSpots) const override { return 0.;};
};

class GeometricBasketCallPayoff : public Payoff {
private:
    double strike_;

public:
    explicit GeometricBasketCallPayoff(double strike) : strike_(strike) {}

    double operator()(const Eigen::VectorXd& terminalSpots) const override {
        const size_t n = static_cast<size_t>(terminalSpots.size());
        double logSum = 0.0;
        for (size_t i = 0; i < n; ++i) {
            logSum += std::log(terminalSpots[i]);
        }
        double geomMean = std::exp(logSum / static_cast<double>(n));
        return std::max(geomMean - strike_, 0.0);
    }
	double evaluate(const PathMatrix& path, const Eigen::VectorXd& initialSpots) const override { return 0.;};
};

class WorstOfAutocallablePayoff : public Payoff {
private:
    std::vector<double> observationTimes_; // e.g., {0.25, 0.5, 0.75, 1.0}
    double autocallBarrier_;                // e.g., 1.0 (100% of initial)
    double protectionBarrier_;             // e.g., 0.60 (60% of initial)
    double couponRatePerYear_;             // e.g., 0.08 (8% per annum)
    double nominal_;                       // e.g., 100.0

public:
    WorstOfAutocallablePayoff(const std::vector<double>& obsTimes,
                              double autocallBarrier,
                              double protectionBarrier,
                              double couponRate,
                              double nominal = 100.0)
        : observationTimes_(obsTimes),
          autocallBarrier_(autocallBarrier),
          protectionBarrier_(protectionBarrier),
          couponRatePerYear_(couponRate),
          nominal_(nominal) {}
	double operator()(const Eigen::VectorXd& terminalSpots) const override { return 0.;}
    double evaluate(const PathMatrix& path, const Eigen::VectorXd& initialSpots) const override {
        size_t numObs = observationTimes_.size();
        size_t nAssets = initialSpots.size();

        // Check each observation date for an early autocall event
        for (size_t step = 0; step < numObs; ++step) {
            const auto& spotsAtT = path[step + 1]; // path[0] is t=0
            double t = observationTimes_[step];

            // Find the worst-performing asset ratio: min_i (S_i(t) / S_i(0))
            double worstRatio = 1.0;
            for (size_t i = 0; i < nAssets; ++i) {
                double ratio = spotsAtT[i] / initialSpots[i];
                if (ratio < worstRatio) {
                    worstRatio = ratio;
                }
            }

            // If the worst asset is above the autocall barrier, product redeems early!
            if (worstRatio >= autocallBarrier_) {
                double coupon = nominal_ * (1.0 + couponRatePerYear_ * t);
                return coupon;
            }
        }

        // If it never autocalled, observe final asset state at maturity
        const auto& finalSpots = path.back();
        double finalWorstRatio = 1.0;
        bool barrierBreached = false;

        // Check if protection barrier was breached across any monitoring date
        for (size_t step = 0; step < numObs; ++step) {
            const auto& spotsAtT = path[step + 1];
            for (size_t i = 0; i < nAssets; ++i) {
                if ((spotsAtT[i] / initialSpots[i]) < protectionBarrier_) {
                    barrierBreached = true;
                }
            }
        }

        for (size_t i = 0; i < nAssets; ++i) {
            double ratio = finalSpots[i] / initialSpots[i];
            if (ratio < finalWorstRatio) {
                finalWorstRatio = ratio;
            }
        }

        // Payoff at maturity
        if (!barrierBreached) {
            return nominal_; // Full principal protection
        } else {
            return nominal_ * finalWorstRatio; // Participation in worst-asset drop
        }
    }
};

} // namespace pricer
