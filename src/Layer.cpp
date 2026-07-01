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