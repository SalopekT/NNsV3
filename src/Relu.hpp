#ifndef RELU
#define RELU
#include "Activation.hpp"

class Relu : public Activation{
    public:
        Eigen::VectorXd calculate(const Eigen::VectorXd& input) override;
};

#endif
