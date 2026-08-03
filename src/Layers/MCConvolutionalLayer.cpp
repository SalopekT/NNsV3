#include "MCConvolutionalLayer.hpp"

MCConvolutionalLayer::MCConvolutionalLayer(int inputChannels, int outputChannels, int dimensionInput, int dimensionKernel) 
                                                                        : Layer(dimensionInput,dimensionInput), inputChannels(inputChannels), 
                                                                                outputChannels(outputChannels){
    for (int i=0;i<outputChannels;i++){
        this->filters.push_back(std::make_shared<ConvolutionalLayer>(dimensionInput,dimensionKernel));
    }

}

Eigen::VectorXd MCConvolutionalLayer::simpleCalculateOutput(const Eigen::VectorXd& input){
    Eigen::VectorXd allChannels = Eigen::VectorXd::Zero(dimensionOutput * outputChannels);
    
    for (int i = 0; i < outputChannels; i++) {
        // Sum all input channels into a single channel first
        // This ensures filters.at(i) receives and caches ONE valid input vector per forward pass
        Eigen::VectorXd summedInput = Eigen::VectorXd::Zero(dimensionInput);
        for (int j = 0; j < inputChannels; j++) {
            summedInput += input.segment(j * dimensionInput, dimensionInput);
        }
        
        // Pass the combined input into the shared filter
        Eigen::VectorXd filterOutput = this->filters.at(i)->simpleCalculateOutput(summedInput);
        
        int outOffset = i * this->dimensionOutput;
        allChannels.segment(outOffset, dimensionOutput) = filterOutput;
    }
    return allChannels;
}
 Eigen::MatrixXd MCConvolutionalLayer::calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){
    for (int i=0;i<outputChannels;i++){
        int firstElement = i*this->dimensionOutput;
        Eigen::VectorXd currentChannelAdjoint = adjointPrev.segment(firstElement, dimensionOutput);
        this->filters.at(i)->calculateAdjointWeights(currentChannelAdjoint);
    }
    return this->filters.at(0)->getAdjointWeights(); //this return doesn't really matter
 }
        
Eigen::VectorXd MCConvolutionalLayer::calculateAdjointInput(const Eigen::VectorXd& adjointPrev){
    Eigen::VectorXd allInputAdjoint = Eigen::VectorXd::Zero(dimensionInput * inputChannels);

    for (int i = 0; i < outputChannels; i++) {
        int outOffset = i * this->dimensionOutput;
        Eigen::VectorXd currentChannelAdjoint = adjointPrev.segment(outOffset, dimensionOutput);
        
        Eigen::VectorXd singleInputAdjoint = this->filters.at(i)->calculateAdjointInput(currentChannelAdjoint);
        
        for (int j = 0; j < inputChannels; j++) {
            int inOffset = j * this->dimensionInput;
            allInputAdjoint.segment(inOffset, dimensionInput) += singleInputAdjoint;
        }
    }
    return allInputAdjoint;
}

void MCConvolutionalLayer::resetAdjointInput(){
    this->adjointInput = Eigen::VectorXd::Zero(dimensionInput * inputChannels);
}

void MCConvolutionalLayer::resetAdjointWeights(){
    // MCConvolutionalLayer doesn't store weights - they're in the filters
    // So we reset the filters' adjoints instead
    for (int i = 0; i < outputChannels; i++) {
        this->filters.at(i)->resetAdjointWeights();
    }
}

void MCConvolutionalLayer::resetCumulativeAdjointWeights(){
    // MCConvolutionalLayer doesn't store weights - they're in the filters
    for (int i = 0; i < outputChannels; i++) {
        this->filters.at(i)->resetCumulativeAdjointWeights();
    }
}