#ifndef CE_LOSS
#define CE_LOSS

#include <Eigen/Dense>
#include <iostream>
#include "Loss.hpp"

class CrossEntropyLoss : public Loss{
    public:
        double calculate(const Eigen::VectorXd& input, const Eigen::VectorXd& target) override;
        Eigen::VectorXd getAdjoint(const Eigen::VectorXd& input, const Eigen::VectorXd& target) override;
};


#endif