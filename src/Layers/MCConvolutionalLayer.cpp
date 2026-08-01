#include "MCConvolutionalLayer.hpp"

MCConvolutionalLayer::MCConvolutionalLayer(int inputChannels, int outputChannels, int dimensionInput, int dimensionKernel) 
                                                                        : Layer(dimensionInput,dimensionInput), inputChannels(inputChannels), 
                                                                                outputChannels(outputChannels){
    for (int i=0;i<outputChannels;i++){
        this->filters.push_back(std::make_shared<ConvolutionalLayer>(dimensionInput,dimensionKernel));
    }

}

Eigen::VectorXd MCConvolutionalLayer::simpleCalculateOutput(const Eigen::VectorXd& input){
    Eigen::VectorXd allChannels = Eigen::VectorXd::Zero(dimensionOutput*outputChannels);
    for (int i=0;i<outputChannels;i++){
        Eigen::VectorXd oneFilterOutput = Eigen::VectorXd::Zero(dimensionOutput);
        for (int j=0;j<inputChannels;j++){
            int firstElement = j*this->dimensionInput;
            Eigen::VectorXd currentChannel = input.segment(firstElement, dimensionInput);
            Eigen::VectorXd oneChannelOutput = this->filters.at(i)->simpleCalculateOutput(currentChannel);
            oneFilterOutput+=oneChannelOutput;
        }
        int firstElement = i*this->dimensionOutput;
        allChannels.segment(firstElement, dimensionOutput) = oneFilterOutput;
        
    }
   
}
 Eigen::MatrixXd calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){

 }
        
 Eigen::VectorXd calculateAdjointInput(const Eigen::VectorXd& adjointPrev){

}