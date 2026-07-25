#include "MCConvolutionalLayer.hpp"

MCConvolutionalLayer::MCConvolutionalLayer(int inputChannels, int outputChannels, int dimensionInput, int dimensionKernel) 
                                                                        : Layer(dimensionInput,dimensionInput), inputChannels(inputChannels), 
                                                                                outputChannels(outputChannels){
    for (int i=0;i<outputChannels;i++){
        this->filters.push_back(std::make_shared<ConvolutionalLayer>(dimensionInput,dimensionKernel));
    }

}

Eigen::VectorXd MCConvolutionalLayer::simpleCalculateOutput(const Eigen::VectorXd& input){

}
 Eigen::MatrixXd calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){

 }
        
 Eigen::VectorXd calculateAdjointInput(const Eigen::VectorXd& adjointPrev){

}