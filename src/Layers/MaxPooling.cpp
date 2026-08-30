#include "MaxPooling.hpp"


MaxPooling::MaxPooling(int dimensionInput, int dimensionOutput, int kernelSize) : Layer(dimensionInput, dimensionOutput), kernelSize(kernelSize){
    int width = std::sqrt(dimensionInput);
    int height = std::sqrt(dimensionInput);
    for (int i=0;i<height;i+=kernelSize){
        for (int j=0;j<width;j+=2){
            int firstKernelPosition = i*width+j;
            Eigen::MatrixXi currentIndices = Eigen::MatrixXi::Zero(kernelSize,kernelSize);
            currentIndices(0,0) = firstKernelPosition;
            for (int k=0;k<kernelSize;k++){
                for (int s=0;s<kernelSize;s++){
                    currentIndices(k,s) = firstKernelPosition;
                }
            }
        }
    }
};


Eigen::VectorXd MaxPooling::simpleCalculateOutput(const Eigen::VectorXd& input){
    Eigen::VectorXd result = Eigen::VectorXd::Zero(dimensionOutput);
    int helper = this->kernelSize/2;
    for (int i=helper;i<dimensionInput-helper;i++){
        
    }
};
        
Eigen::MatrixXd MaxPooling::calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){

};
        
Eigen::VectorXd MaxPooling::calculateAdjointInput(const Eigen::VectorXd& adjointPrev){


};