#ifndef RELU
#define RELU
#include "Activation.hpp"

class Relu : public Activation{
    public:
        Relu(int dimension) : Activation(dimension){}
        Eigen::VectorXd calculate(const Eigen::VectorXd& input) override;
        Eigen::VectorXd getAdjoint(const Eigen::VectorXd& input, const Eigen::VectorXd& prevAdjoint) override;
};

#endif
