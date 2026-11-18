// include/pricer/Payoff.hpp
#pragma once
#include <Eigen/Dense>
#include <algorithm>

namespace pricer {

class Payoff {
public:
    virtual ~Payoff() = default;
    // Evaluates payoff given asset prices at maturity S_T
    virtual double operator()(const Eigen::VectorXd& terminalSpots) const = 0;
};

// Exchange Option: max(S_1(T) - S_2(T), 0)
class MargrabeExchangePayoff : public Payoff {
public:
    double operator()(const Eigen::VectorXd& terminalSpots) const override {
        return std::max(terminalSpots[0] - terminalSpots[1], 0.0);
    }
};

class EuropeanCallPayoff : public Payoff {
private:
    double strike_;
public:
    explicit EuropeanCallPayoff(double strike) : strike_(strike) {}
    
    double operator()(const Eigen::VectorXd& terminalSpots) const override {
        return std::max(terminalSpots[0] - strike_, 0.0);
    }
};

class EuropeanPutPayoff : public Payoff {
private:
    double strike_;
public:
    explicit EuropeanPutPayoff(double strike) : strike_(strike) {}
    
    double operator()(const Eigen::VectorXd& terminalSpots) const override {
        return std::max(strike_ - terminalSpots[0], 0.0);
    }
};

} // namespace pricer
