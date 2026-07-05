#include "LinearLayer.hpp"
#include <random>
#include <iostream>

LinearLayer::LinearLayer(int dimensionInput, int dimensionOutput) : Layer(dimensionInput,dimensionOutput){
    std::mt19937 generator(std::random_device{}());

    double stddev = std::sqrt(2.0 / dimensionInput);
    std::normal_distribution<double> distribution(0.0, stddev);

    weights = Eigen::MatrixXd::Zero(dimensionOutput, dimensionInput + 1);

    weights = weights.unaryExpr([&](double) {
        return distribution(generator);
    });

    adjointWeights = Eigen::MatrixXd::Zero(dimensionOutput, dimensionInput+1);
    adjointInput = Eigen::VectorXd::Zero(dimensionInput);
    cumulativeAdjointWeights = Eigen::MatrixXd::Zero(dimensionOutput, dimensionInput+1);
}

Eigen::VectorXd LinearLayer::simpleCalculateOutput(const Eigen::VectorXd& input_old){
    Eigen::VectorXd input(input_old.size() + 1);
    for (int i=0;i<input_old.size();i++){
        input(i) = input_old(i);
    }
    input(input_old.size())=1; 
    this->input = input;
    Eigen::VectorXd outputBefActivation = this->weights * input;
    Eigen::VectorXd outputAftActivation = outputBefActivation;

    return outputAftActivation;
}



Eigen::MatrixXd LinearLayer::calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){
    for (int i = 0; i < this->dimensionOutput; i++) {
        this->adjointWeights.row(i) += adjointPrev(i) * this->input.transpose();
    }
    return this->adjointWeights;
}


   
   
Eigen::VectorXd LinearLayer::calculateAdjointInput(const Eigen::VectorXd& adjointPrev){
    for (int i=0;i< this->dimensionInput;i++){
        this->adjointInput(i) += weights.col(i).dot(adjointPrev);
    }
    return this->adjointInput;
}

LinearLayer::~LinearLayer(){}