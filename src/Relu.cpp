#include "Relu.hpp"

Eigen::VectorXd Relu::calculate(const Eigen::VectorXd& input){
    return input.array().max(0.0);
}

Eigen::VectorXd Relu::getAdjoint(const Eigen::VectorXd& input, const Eigen::VectorXd& prevAdjoint){
    for (int i=0;i<this->dimension;i++){
        if (input(i)>0){
            this->adjoint(i) += prevAdjoint(i);
        }
    }
}