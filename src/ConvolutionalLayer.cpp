#include "ConvolutionalLayer.hpp"
#include <random>
#include <iostream>

ConvolutionalLayer::ConvolutionalLayer(int dimensionInput, int dimensionKernel) : Layer(dimensionInput, dimensionInput){
    this->dimensionKernel = dimensionKernel;
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
    int widthWithPadding = width+helper*2;
    int heightWithPadding = height+helper*2;
    assert(width * height == dimensionInput);
    this->convMatrix = Eigen::MatrixXd::Zero(dimensionInput, (width+2*helper)*(height+2*helper));
    //convMatrix is operation on flattened padded input image so it has different dimensions than input image

    int counter = 0;
    for (int i=helper;i<helper+height;i++){
        for (int j=helper;j<helper+width;j++){
            int positionInInput = i*widthWithPadding+j; //this is the position of the middle slot in the conv kernel so w22 if 3x3 filter
                                                        //w33 if 5x5 ...
            this->convMatrix(counter,positionInInput) = weights(helper,helper);
            //
            Eigen::MatrixXi indicesWeightsOneMap = Eigen::MatrixXi::Zero(dimensionKernel, dimensionKernel);
            indicesWeightsOneMap(helper,helper) = positionInInput;
            //
            for (int k=1;k<=helper;k++){
                this->convMatrix(counter, positionInInput-k*widthWithPadding) = weights(helper-k,helper);
                this->convMatrix(counter, positionInInput+k*widthWithPadding) = weights(helper+k,helper);
                //
                indicesWeightsOneMap(helper-k,helper) = positionInInput-k*widthWithPadding;
                indicesWeightsOneMap(helper+k,helper) = positionInInput+k*widthWithPadding;

                //
                for (int s=1;s<=helper;s++){
                    this->convMatrix(counter, positionInInput-k*widthWithPadding-s) = weights(helper-k,helper-s);
                    this->convMatrix(counter, positionInInput-k*widthWithPadding+s) = weights(helper-k,helper+s);
                    this->convMatrix(counter, positionInInput+k*widthWithPadding-s) = weights(helper+k,helper-s);
                    this->convMatrix(counter, positionInInput+k*widthWithPadding+s) = weights(helper+k,helper+s);

                    //
                    indicesWeightsOneMap(helper-k,helper-s) = positionInInput-k*widthWithPadding-s;
                    indicesWeightsOneMap(helper-k,helper+s) = positionInInput-k*widthWithPadding+s;
                    indicesWeightsOneMap(helper+k,helper-s) = positionInInput+k*widthWithPadding-s;
                    indicesWeightsOneMap(helper+k,helper+s) = positionInInput+k*widthWithPadding+s;

                    //
                }
            }
        
            
            
            for (int k=1;k<=helper;k++){
                /*std::cout << "k=" << k
              << " counter=" << counter
              << " position=" << positionInInput
              << " weights+col=" << helper+k
              << " weights-col=" << helper-k
              << std::endl;*/
                this->convMatrix(counter,positionInInput+k) = weights(helper,helper+k);
                this->convMatrix(counter,positionInInput-k) = weights(helper,helper-k);

                //
                indicesWeightsOneMap(helper,helper-k) = positionInInput-k;
                indicesWeightsOneMap(helper,helper+k) = positionInInput+k;
                //std::cout << k << std::endl;
            }
            this->indicesWeights.push_back(indicesWeightsOneMap);
            counter++;
        }
        
    }
}
    


Eigen::VectorXd ConvolutionalLayer::simpleCalculateOutput(const Eigen::VectorXd& input){
    //first padding the input
    int helper = dimensionKernel%2;
    int paddingSize = 4*(dimensionKernel+helper*helper);
    int dimensionPaddedInputMatrix = (int) std::sqrt(input.size()+paddingSize);
    Eigen::MatrixXd paddedInputMatrix = Eigen::MatrixXd::Zero(dimensionPaddedInputMatrix,dimensionPaddedInputMatrix);
    int counter = 0;
    for (int i=0;i<dimensionPaddedInputMatrix;i++){
        for (int j=0;j<dimensionPaddedInputMatrix;j++){
            if (i!=0 && i!=dimensionPaddedInputMatrix-1 && j!=0 && j!=dimensionPaddedInputMatrix-1){
                paddedInputMatrix(i,j) = input(counter);
                counter++;
            }
        }
    }
    
    Eigen::VectorXd paddedInput = Eigen::VectorXd::Zero(dimensionPaddedInputMatrix*dimensionPaddedInputMatrix);
    for (int i=0;i<dimensionPaddedInputMatrix;i++){
        for (int j=0;j<dimensionPaddedInputMatrix;j++){
            paddedInput(i*dimensionPaddedInputMatrix+j)=paddedInputMatrix(i,j);
        }
    }
    Eigen::VectorXd outputVec = this->convMatrix * paddedInput;
    return outputVec;
}

Eigen::MatrixXd ConvolutionalLayer::calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){
    for (int i=0;i<this->dimensionInput;i++){
        Eigen::MatrixXi currIndices = this->indicesWeights.at(i);
        for (int j=0;j<dimensionKernel;j++){
            for (int k=0;k<dimensionKernel;k++){
                this->adjointWeights(j,k)+=this->input(currIndices(j,k))*adjointPrev(i);
            }
            
        }
    }
}

Eigen::VectorXd ConvolutionalLayer::calculateAdjointInput(const Eigen::VectorXd& adjointPrev){
    return Eigen::VectorXd::Zero(1);
}

void ConvolutionalLayer::printConvMatrix(){
    std::cout << this->convMatrix << std::endl;
}



std::vector<Eigen::MatrixXi> ConvolutionalLayer::getRowIndices(){
    return this->indicesWeights;
}