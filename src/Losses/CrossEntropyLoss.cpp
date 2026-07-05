#include "CrossEntropyLoss.hpp"
#include <cmath>

double CrossEntropyLoss::calculate(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target){
    this->input = predicted;
    double ce_loss = 0;
    const double eps = 1e-12;
    for (int i=0;i<this->dimensionInput;i++){
        double p = std::max(eps,this->input(i));
        ce_loss -= target(i)*log(p);
    }
    return ce_loss;
}

Eigen::VectorXd CrossEntropyLoss::calculateAdjoint(const Eigen::VectorXd& target){
    for (int i=0;i<dimensionInput;i++){
        this->adjoint(i) = this->input(i)-target(i);
        // this->adjoint(i) = target(i)/this->input(i);
    }
    return this->adjoint;
}