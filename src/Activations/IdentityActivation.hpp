#ifndef IDENTITY_ACTIVATION
#define IDENTITY_ACTIVATION

#include "Activation.hpp"

class IdentityActivation : public Activation {
public:
    IdentityActivation(int dimension) : Activation(dimension) {};
    Eigen::VectorXd calculate(const Eigen::VectorXd& input) override {
        this->input = input;
        return input;
    }

    Eigen::VectorXd calculateAdjoint(const Eigen::VectorXd& prevAdjoint) override {
        this->adjoint = prevAdjoint;
        return this->adjoint;
    }

    virtual ~IdentityActivation() {};
};

#endif