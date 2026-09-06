#ifndef MAX_POOL
#define MAX_POOL
#include "Layer.hpp"
#include <vector>   
class MaxPooling : public Layer{
    private:
        int kernelSize;
        std::vector<Eigen::MatrixXi> indicesWeights;
        std::vector<bool> isMaxIndex;
    public:
        MaxPooling(int dimensionInput, int dimensionOutput, int kernelSize);
        Eigen::VectorXd simpleCalculateOutput(const Eigen::VectorXd& input) override;
        Eigen::MatrixXd calculateAdjointWeights(const Eigen::VectorXd& adjointPrev) override;
        Eigen::VectorXd calculateAdjointInput(const Eigen::VectorXd& adjointPrev) override;
};

#endif