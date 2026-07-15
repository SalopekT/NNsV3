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

            for (int k=1;k<=helper;k++){
                this->convMatrix(counter, positionInInput-k*widthWithPadding) = weights(helper-k,helper);
                this->convMatrix(counter, positionInInput+k*widthWithPadding) = weights(helper+k,helper);
                for (int s=1;s<=helper;s++){
                    this->convMatrix(counter, positionInInput-k*widthWithPadding-s) = weights(helper-k,helper-s);
                    this->convMatrix(counter, positionInInput-k*widthWithPadding+s) = weights(helper-k,helper+s);
                    this->convMatrix(counter, positionInInput+k*widthWithPadding-s) = weights(helper+k,helper-s);
                    this->convMatrix(counter, positionInInput+k*widthWithPadding+s) = weights(helper+k,helper+s);
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
                //std::cout << k << std::endl;
            }
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
}

Eigen::MatrixXd ConvolutionalLayer::calculateAdjointWeights(const Eigen::VectorXd& adjointPrev){
    return Eigen::VectorXd::Zero(1);
}

Eigen::VectorXd ConvolutionalLayer::calculateAdjointInput(const Eigen::VectorXd& adjointPrev){
    return Eigen::VectorXd::Zero(1);
}

void ConvolutionalLayer::printConvMatrix(){
    std::cout << this->convMatrix << std::endl;
}