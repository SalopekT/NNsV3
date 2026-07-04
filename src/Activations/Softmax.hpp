#ifndef SOFTMAX
#define SOFTMAX
#include "Activation.hpp"

class Softmax : public Activation{
    public:
        Softmax(int dimension) : Activation(dimension){}
        Eigen::VectorXd calculate(const Eigen::VectorXd& input) override;
        Eigen::VectorXd calculateAdjoint(const Eigen::VectorXd& prevAdjoint) override;
};

#endif
