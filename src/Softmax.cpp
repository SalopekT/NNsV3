#include "Softmax.hpp"

Eigen::VectorXd Softmax::calculate(const Eigen::VectorXd& input){
    //subtracting maxVal makes it stable (there are no large exponents)
    double maxVal = input.maxCoeff();
    Eigen::VectorXd expVec = input.array()
        .unaryExpr([maxVal](double x){
            return std::exp(x - maxVal);
        });

    double sum = expVec.sum();

    Eigen::VectorXd outputAftActivation = expVec / sum;
    return outputAftActivation;
}

Eigen::VectorXd Softmax::getAdjoint(const Eigen::VectorXd& input, const Eigen::VectorXd& prevAdjoint){
    //subtracting maxVal makes it stable (there are no large exponents)
    double maxVal = input.maxCoeff();
    Eigen::VectorXd expVec = input.array()
        .unaryExpr([maxVal](double x){
            return std::exp(x - maxVal);
        });

    double sum = expVec.sum();
    double sum_squared = sum*sum;
    for (int i=0;i<this->dimension;i++){
        this->adjoint(i)+=((expVec(i)*sum-expVec(i)*expVec(i))/sum_squared)*prevAdjoint(i);
        for (int j=0;j<dimension;j++){
            if (i!=j){
                this->adjoint(i)-=(expVec(i)*expVec(j)/sum_squared)*prevAdjoint(j);
            }
        }
    }
    return this->adjoint;
}