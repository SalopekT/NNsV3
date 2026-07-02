#include "CrossEntropyLoss.hpp"
#include <cmath>

double CrossEntropyLoss::calculate(const Eigen::VectorXd& input, const Eigen::VectorXd& target){
    double ce_loss = 0;
    for (int i=0;i<this->dimensionInput;i++){
        ce_loss -= target(i)*log(input(i));
    }
    return ce_loss;
}

Eigen::VectorXd CrossEntropyLoss::getAdjoint(const Eigen::VectorXd& input, const Eigen::VectorXd& target){
    for (int i=0;i<dimensionInput;i++){
        this->adjoint(i)+=target(i)/input(i);
    }
    return this->adjoint;
}