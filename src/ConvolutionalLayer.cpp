#include "ConvolutionalLayer.hpp"
#include <random>
#include <iostream>

ConvolutionalLayer::ConvolutionalLayer(int dimensionInput, int dimensionKernel) : Layer(dimensionInput, dimensionInput){
    std::mt19937 generator(std::random_device{}());

    double stddev = std::sqrt(2.0 / dimensionKernel*dimensionKernel);
    std::normal_distribution<double> distribution(0.0, stddev);

    weights = Eigen::MatrixXd::Zero(dimensionKernel, dimensionKernel);

    weights = weights.unaryExpr([&](double) {
        return distribution(generator);
    });

    adjointWeights = Eigen::MatrixXd::Zero(dimensionKernel, dimensionKernel);
    adjointInput = Eigen::VectorXd::Zero(dimensionInput);
    cumulativeAdjointWeights = Eigen::MatrixXd::Zero(dimensionKernel, dimensionKernel);

    int helper = dimensionKernel%2;
    int width = std::sqrt(dimensionInput);
    int height = std::sqrt(dimensionInput);
    assert(width * width == dimensionInput);
    this->convMatrix = Eigen::MatrixXd::Zero(dimensionInput, (dimensionInput+helper)*(dimensionInput+helper));

    for (int i=helper;i<helper+height;i++){
        for (int j=helper;j<helper+width;j++){
            
        }
    }

}