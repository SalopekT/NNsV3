#ifndef MCCONV_LAYER
#define MCCONV_LAYER

#include "Layer.hpp"
#include "ConvolutionalLayer.hpp"
#include <memory>

class MCConvolutionalLayer : public Layer{
    private:
        int inputChannels;
        int outputChannels;

        std::vector<std::shared_ptr<ConvolutionalLayer>> filters; //this is actually 2d array of filters
    public:
        //dimension output here is still dimension of one channel convolution
        MCConvolutionalLayer(int inputChannels, int outputChannels, int dimensionInput, int dimensionKernel);

        Eigen::VectorXd simpleCalculateOutput(const Eigen::VectorXd& input) override;
        Eigen::MatrixXd calculateAdjointWeights(const Eigen::VectorXd& adjointPrev) override;
        Eigen::VectorXd calculateAdjointInput(const Eigen::VectorXd& adjointPrev) override;

        virtual void resetAdjointInput() override;
        virtual void resetAdjointWeights() override;
        virtual void resetCumulativeAdjointWeights() override;


};
#endif