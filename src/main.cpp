#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include "Layer.hpp"
#include "LinearLayer.hpp"
#include "Activation.hpp"
#include "Relu.hpp"
#include "Network.hpp"

int main() {
    std::unique_ptr<Layer> l1 = std::make_unique<LinearLayer>(4,3);
    std::unique_ptr<Activation> a1 = std::make_unique<Relu>(3);
    Network* net = new Network();
    net->addLayerAndActivation(std::move(l1),std::move(a1));

    Eigen::Vector4d input;
    input << 1.0, 2.0, 3.0, 4.0;
    l1->printWeights();
    std::cout << "---------\n";
    Eigen::VectorXd result = net->forwardPass(input);
    std::cout << result << std::endl;

    delete net;
    return 0;
}