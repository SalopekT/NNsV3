#ifndef SOFTMAX
#define SOFTMAX
#include "Activation.hpp"

class Softmax : public Activation{
    public:
        Eigen::VectorXd calculate(const Eigen::VectorXd& input) override;
        Eigen::VectorXd getAdjoint(const Eigen::VectorXd& input, const Eigen::VectorXd& prevAdjoint) override;
};

#endif
