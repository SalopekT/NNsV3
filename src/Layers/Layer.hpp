#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include <iostream>
class Layer{
    protected:
        Eigen::MatrixXd weights; //biases are embedded in the last column
        Eigen::MatrixXd adjointWeights; //jacobian of output w.r.t weights
        Eigen::MatrixXd adjointInput; //jacobian of output w.r.t input
        int dimensionInput,dimensionOutput;
        Eigen::VectorXd input;

        //this is for batch gradient descent
        Eigen::MatrixXd cumulativeAdjointWeights;
    public:
        Layer(int dimensionInput, int dimensionOutput) : dimensionInput(dimensionInput), dimensionOutput(dimensionOutput){}
        Layer(int dimensionInput, int dimensionOutput, Eigen::MatrixXd weights) : dimensionInput(dimensionInput), dimensionOutput(dimensionOutput), weights(weights) {}
        virtual Eigen::VectorXd simpleCalculateOutput(const Eigen::VectorXd& input) = 0;
        virtual Eigen::MatrixXd calculateAdjointWeights(const Eigen::VectorXd& adjointPrev) = 0;
        virtual Eigen::VectorXd calculateAdjointInput(const Eigen::VectorXd& adjointPrev) = 0;

        Eigen::MatrixXd getAdjointWeights();
        Eigen::VectorXd getAdjointInput();
        void resetAdjointWeights();
        void resetAdjointInput();
        void resetCumulativeAdjointWeights();

        void updateWeights(double learningRate);
        void storeWeightsCumulative();
        void updateWeightsBatch(double learningrate, int batchSize);

        virtual ~Layer() {};
        Eigen::VectorXd getInput();
        int getDimensionInput();
        int getDimensionOutput();

        double getWeight(int row, int col);
        void setWeight(int row, int col, double value);

        Eigen::MatrixXd getWeights() { return weights; }
        void setWeights(const Eigen::MatrixXd& newWeights) { weights = newWeights; }

        void printWeights();
        
};



#endif