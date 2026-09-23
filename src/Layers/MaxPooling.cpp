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
    
    assert(this->indicesWeights.size() == dimensionOutput);
    isMaxIndex.resize(dimensionInput, false);
};


Eigen::VectorXd MaxPooling::simpleCalculateOutput(const Eigen::VectorXd& input){
    this->isMaxIndex.clear();
    this->isMaxIndex.resize(dimensionInput, false);

    this->input = input;
    int width = std::sqrt(dimensionInput);
    Eigen::VectorXd result = Eigen::VectorXd::Zero(dimensionOutput);
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
        isMaxIndex[indicesWeights.at(i)(maxI,maxJ)] = true; 

    }

    return result;
    
};
        
Eigen::MatrixXd MaxPooling::calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){
    //there are no weights so nothing to optimize here
    return Eigen::MatrixXd::Zero(dimensionOutput,dimensionInput);
};
        
Eigen::VectorXd MaxPooling::calculateAdjointInput(const Eigen::VectorXd& adjointPrev){
    //counting number of true values in isMaxIndex
    int maxCount = 0;
    for (int i = 0; i < this->dimensionInput; i++){
        if (this->isMaxIndex.at(i)) maxCount++;
    }
    
    if (maxCount != adjointPrev.size()) {
        throw std::runtime_error(
            "Size mismatch in calculateAdjointInput: " + 
            std::to_string(maxCount) + " max indices but adjointPrev size is " + 
            std::to_string(adjointPrev.size())
        );
    }
    
    this->adjointInput = Eigen::VectorXd::Zero(dimensionInput);
    int counter = 0;
    for (int i=0;i<this->dimensionInput;i++){
        if (this->isMaxIndex.at(i)){
            this->adjointInput(i)=adjointPrev(counter);
            counter++;
        }
        else this->adjointInput(i) = 0;
    }
    return this->adjointInput;

};

 void MaxPooling::resetAdjointWeights(){
    return;
 };


 void MaxPooling::resetCumulativeAdjointWeights(){
    return;
 };