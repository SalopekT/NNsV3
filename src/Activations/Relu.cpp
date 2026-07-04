#include "Relu.hpp"

Eigen::VectorXd Relu::calculate(const Eigen::VectorXd& input){
    this->input=input;
    return input.array().max(0.0);
}

Eigen::VectorXd Relu::calculateAdjoint(const Eigen::VectorXd& prevAdjoint){
    for (int i=0;i<this->dimension;i++){
        if (this->input(i)>0){
            this->adjoint(i) += prevAdjoint(i);
        }
    }
    return this->adjoint;
}