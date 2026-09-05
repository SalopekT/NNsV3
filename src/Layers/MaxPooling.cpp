#include "MaxPooling.hpp"


MaxPooling::MaxPooling(int dimensionInput, int dimensionOutput, int kernelSize) : Layer(dimensionInput, dimensionOutput), kernelSize(kernelSize){
    int width = std::sqrt(dimensionInput);
    int height = std::sqrt(dimensionInput);
    for (int i=0;i<=height-kernelSize;i+=kernelSize){
        for (int j=0;j<=width-kernelSize;j+=kernelSize){
            int firstKernelPosition = i*width+j;
            Eigen::MatrixXi currentIndices = Eigen::MatrixXi::Zero(kernelSize,kernelSize);
            currentIndices(0,0) = firstKernelPosition;
            for (int k=0;k<kernelSize;k++){
                for (int s=0;s<kernelSize;s++){
                    int kernelPosition = firstKernelPosition+k*width+s;
                    currentIndices(k,s) = kernelPosition;
                }
            }

            this->indicesWeights.push_back(currentIndices);
        }
    }
    std::cout << this->indicesWeights.size() <<std::endl;
    assert(this->indicesWeights.size() == dimensionOutput);
    isMaxIndex.resize(dimensionInput, false);
};


Eigen::VectorXd MaxPooling::simpleCalculateOutput(const Eigen::VectorXd& input){
    int width = std::sqrt(dimensionInput);
    Eigen::VectorXd result = Eigen::VectorXd::Zero(dimensionOutput);
    int outputIndex = 0;
    //here i can just go through all the indicesWeights i stored in constructor
    for (int i=0;i<this->indicesWeights.size();i++){
        Eigen::MatrixXd currentInputs = Eigen::MatrixXd::Zero(kernelSize,kernelSize);
        for (int j=0;j<kernelSize;j++){
            for (int k=0;k<kernelSize;k++){
                currentInputs(j,k) = input(indicesWeights.at(i)(j,k));
            }
        }
        Eigen::Index maxI, maxJ;
        double maxValue = currentInputs.maxCoeff(&maxI,&maxJ);
        result(i) = maxValue;
        isMaxIndex[maxI*width+maxJ] = true; 
    }

    return result;
    
};
        
Eigen::MatrixXd MaxPooling::calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){
    //there are no weights so nothing to optimize here
    return Eigen::VectorXd::Zero(dimensionOutput);
};
        
Eigen::VectorXd MaxPooling::calculateAdjointInput(const Eigen::VectorXd& adjointPrev){
    for (int i=0;i<this->dimensionInput;i++){
        if (this->isMaxIndex.at(i)) this->adjointInput(i)=adjointPrev(i);
        else this->adjointInput(i) = 0;
    }
    return this->adjointInput;

};