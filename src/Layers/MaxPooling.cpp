#include "MaxPooling.hpp"

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