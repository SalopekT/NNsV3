#ifndef MAX_POOL
#define MAX_POOL
#include "Layer.hpp"
class MaxPooling : public Layer{
    private:
        int kernelSize;
    public:
        MaxPooling(int dimensionInput, int dimensionOutput, int kernelSize) : Layer(dimensionInput, dimensionOutput), kernelSize(kernelSize){};
        Eigen::VectorXd simpleCalculateOutput(const Eigen::VectorXd& input) override;
        Eigen::MatrixXd calculateAdjointWeights(const Eigen::VectorXd& adjointPrev) override;
        Eigen::VectorXd calculateAdjointInput(const Eigen::VectorXd& adjointPrev) override;
};

#endif