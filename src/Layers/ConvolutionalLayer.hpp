#ifndef CONVOLUTIONAL
#define CONVOLUTIONAL
#include "Layer.hpp"
#include <vector>
class ConvolutionalLayer : public Layer{ //it inherits Layer so here: weights matrix is a conv kernel so e.g. 3x3
    private:                        // and convMatrix is a sparse matrix equal to applying kernel to input
        Eigen::MatrixXd convMatrix;
        int dimensionKernel;

        std::vector<Eigen::MatrixXi> indicesWeights; //this stores for each row in convMatrix which padded input gets multiplied 

    public:
        ConvolutionalLayer(int dimensionInput, int dimensionKernel);
        ConvolutionalLayer(int dimensionInput, const Eigen::MatrixXd& weights)
        : Layer(dimensionInput,dimensionInput,weights){}

        Eigen::VectorXd simpleCalculateOutput(const Eigen::VectorXd& input) override;
        Eigen::MatrixXd calculateAdjointWeights(const Eigen::VectorXd& adjointPrev) override;
        Eigen::VectorXd calculateAdjointInput(const Eigen::VectorXd& adjointPrev) override;

        void resetAdjointWeights() override;
        void resetCumulativeAdjointWeights() override;

        void printConvMatrix();
        std::vector<Eigen::MatrixXi> getRowIndices();

        ~ConvolutionalLayer(){};
};

#endif;