#ifndef CE_LOSS
#define CE_LOSS

#include <Eigen/Dense>
#include <iostream>
#include "Loss.hpp"

class CrossEntropyLoss : public Loss{
    public:
        CrossEntropyLoss(int dimensionInput) : Loss(dimensionInput){}
        double calculate(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) override;
        Eigen::VectorXd calculateAdjoint(const Eigen::VectorXd& target) override; 
};


#endif