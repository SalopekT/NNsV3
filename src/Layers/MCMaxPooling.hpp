#ifndef MCPOOL_LAYER
#define MCPOOL_LAYER
#include "Layer.hpp"
#include "MaxPooling.hpp"
#include <memory>

class MCMaxPooling : public Layer{
    private:
        int inputChannels;
        int outputChannels;

        std::vector<std::shared_ptr<MaxPooling>> poolingLayers;

    public:
        MCMaxPooling(int inputChannels, int outputChannels, int dimensionInput, int dimensionOutput, int kernelSize);

        Eigen::VectorXd simpleCalculateOutput(const Eigen::VectorXd& input) override;
        Eigen::MatrixXd calculateAdjointWeights(const Eigen::VectorXd& adjointPrev) override;
        Eigen::VectorXd calculateAdjointInput(const Eigen::VectorXd& adjointPrev) override;
};

#endif