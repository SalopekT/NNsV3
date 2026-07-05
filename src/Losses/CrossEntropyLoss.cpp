#include "CrossEntropyLoss.hpp"
#include <cmath>

double CrossEntropyLoss::calculate(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target){
    this->input = predicted;
    double ce_loss = 0;
    for (int i=0;i<this->dimensionInput;i++){
        ce_loss -= target(i)*log(this->input(i));
    }
    return ce_loss;
}

Eigen::VectorXd CrossEntropyLoss::calculateAdjoint(const Eigen::VectorXd& target){
    for (int i=0;i<dimensionInput;i++){
        this->adjoint(i) = this->input(i)-target(i);
    }
    return this->adjoint;
}