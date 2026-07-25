#ifndef LINEAR_LAYER
#define LINEAR_LAYER
#include "Layer.hpp"

class LinearLayer : public Layer{
    public:
        LinearLayer(int dimensionInput, int dimensionOutput);
        LinearLayer(int dimensionInput, int dimensionOutput, const Eigen::MatrixXd& weights)
        : Layer(dimensionInput,dimensionOutput,weights){}

        Eigen::VectorXd simpleCalculateOutput(const Eigen::VectorXd& input) override;
        Eigen::MatrixXd calculateAdjointWeights(const Eigen::VectorXd& adjointPrev) override;
        Eigen::VectorXd calculateAdjointInput(const Eigen::VectorXd& adjointPrev) override;

        ~LinearLayer();
};

#endif