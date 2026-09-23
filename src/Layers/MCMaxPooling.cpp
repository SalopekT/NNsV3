#include "MCMaxPooling.hpp"
#include <memory>

MCMaxPooling::MCMaxPooling(int inputChannels, int outputChannels, int dimensionInput, int dimensionOutput, int kernelSize) : 
                                            Layer(dimensionInput, dimensionOutput), inputChannels(inputChannels), outputChannels(outputChannels){
            

        for (int i=0;i<inputChannels;i++){
            this->poolingLayers.push_back(std::make_shared<MaxPooling>(dimensionInput,dimensionOutput,kernelSize));
        }
}

Eigen::VectorXd MCMaxPooling::simpleCalculateOutput(const Eigen::VectorXd& input){
    Eigen::VectorXd output = Eigen::VectorXd::Zero(dimensionOutput*outputChannels);
    for (int i=0;i<inputChannels;i++){
        Eigen::VectorXd inputSegment = input.segment(i*dimensionInput,dimensionInput);
        Eigen::VectorXd pooledVector = this->poolingLayers.at(i)->simpleCalculateOutput(inputSegment);
        output.segment(i*dimensionOutput,dimensionOutput) = pooledVector;
    }
    return output;
};
        
Eigen::MatrixXd MCMaxPooling::calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){
    return Eigen::MatrixXd::Zero(1,1); //this doesn't matter
        
};

Eigen::VectorXd MCMaxPooling::calculateAdjointInput(const Eigen::VectorXd& adjointPrev){
    Eigen::VectorXd allInputAdjoint = Eigen::VectorXd::Zero(dimensionInput * inputChannels);
    
    for (int i=0; i<inputChannels; i++){
        Eigen::VectorXd adjointSegment = adjointPrev.segment(i*dimensionOutput, dimensionOutput);
        
        Eigen::VectorXd result = this->poolingLayers.at(i)->calculateAdjointInput(adjointSegment);
        allInputAdjoint.segment(i*dimensionInput, dimensionInput) = result;
    }
    
    return allInputAdjoint;
}

 void MCMaxPooling::resetAdjointWeights(){
    return;
 };


 void MCMaxPooling::resetCumulativeAdjointWeights(){
    return;
 };