#include "Layer.hpp"

void Layer::printWeights(){
            std::cout << "Weights: " << std::endl;
            std::cout << weights << std::endl;
            std::cout << "-------------\n";
};



Eigen::VectorXd Layer::getInput(){
    return this->input;
}

int Layer::getDimensionInput(){
    return this->dimensionInput;
}
int Layer::getDimensionOutput(){
    return this->dimensionOutput;
}

double Layer::getWeight(int row, int col){
    return this->weights(row,col);
}

void Layer::setWeight(int row, int col, double value){
    this->weights(row,col) = value;
}

Eigen::MatrixXd Layer::getAdjointWeights(){
    return this->adjointWeights;
}

Eigen::VectorXd Layer::getAdjointInput(){
    return this->adjointInput;
}

void Layer::resetAdjointWeights(){
    this->adjointWeights = Eigen::MatrixXd::Zero(dimensionOutput, dimensionInput+1);
}
void Layer::resetAdjointInput(){
    adjointInput = Eigen::VectorXd::Zero(dimensionInput);
}

void Layer::updateWeights(double learningRate){
    this->weights -= learningRate*this->adjointWeights;
}