#include "MCConvolutionalLayer.hpp"

MCConvolutionalLayer::MCConvolutionalLayer(int inputChannels, int outputChannels, int dimensionInput, int dimensionKernel) 
                                                                        : Layer(dimensionInput,dimensionInput), inputChannels(inputChannels), 
                                                                                outputChannels(outputChannels){
    for (int i = 0; i < outputChannels; i++) {
        for (int j = 0; j < inputChannels; j++) {
            this->filters.push_back(std::make_shared<ConvolutionalLayer>(dimensionInput, dimensionKernel));
        }
    }

}

Eigen::VectorXd MCConvolutionalLayer::simpleCalculateOutput(const Eigen::VectorXd& input){
    Eigen::VectorXd allChannels = Eigen::VectorXd::Zero(dimensionOutput * outputChannels);
    
    for (int i = 0; i < outputChannels; i++) {
        Eigen::VectorXd output = Eigen::VectorXd::Zero(dimensionOutput);
        for (int j = 0; j < inputChannels; j++) {
            Eigen::VectorXd inputChannel = input.segment(j * dimensionInput, dimensionInput);
            Eigen::VectorXd filterOutput = this->filters.at(i * inputChannels + j)->simpleCalculateOutput(inputChannel);
            output += filterOutput;
        }
        
        int outOffset = i * this->dimensionOutput;
        allChannels.segment(outOffset, dimensionOutput) = output;
    }
    return allChannels;
}

 Eigen::MatrixXd MCConvolutionalLayer::calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){
    for (int i = 0; i < outputChannels; i++) {
        int firstElement = i * this->dimensionOutput;
        Eigen::VectorXd currentChannelAdjoint = adjointPrev.segment(firstElement, dimensionOutput);
        
        for (int j = 0; j < inputChannels; j++) {
            this->filters.at(i * inputChannels + j)->calculateAdjointWeights(currentChannelAdjoint);
        }
    }
    return this->filters.at(0)->getAdjointWeights();
}
        
Eigen::VectorXd MCConvolutionalLayer::calculateAdjointInput(const Eigen::VectorXd& adjointPrev){
    Eigen::VectorXd allInputAdjoint = Eigen::VectorXd::Zero(dimensionInput * inputChannels);

    for (int i = 0; i < outputChannels; i++) {
        int outOffset = i * this->dimensionOutput;
        Eigen::VectorXd currentChannelAdjoint = adjointPrev.segment(outOffset, dimensionOutput);
        
        for (int j = 0; j < inputChannels; j++) {
            Eigen::VectorXd inputAdjoint = this->filters.at(i * inputChannels + j)->calculateAdjointInput(currentChannelAdjoint);
            
            int inOffset = j * this->dimensionInput;
            allInputAdjoint.segment(inOffset, dimensionInput) += inputAdjoint;
        }
    }
    return allInputAdjoint;
}

void MCConvolutionalLayer::resetAdjointInput(){
    this->adjointInput = Eigen::VectorXd::Zero(dimensionInput * inputChannels);
}

void MCConvolutionalLayer::resetAdjointWeights(){
    for (int i = 0; i < outputChannels; i++) {
        this->filters.at(i)->resetAdjointWeights();
    }
}

void MCConvolutionalLayer::resetCumulativeAdjointWeights(){
    for (int i = 0; i < outputChannels; i++) {
        this->filters.at(i)->resetCumulativeAdjointWeights();
    }
}