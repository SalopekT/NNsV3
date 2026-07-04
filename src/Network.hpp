#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <iostream>
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "Layer.hpp"
#include "Activations/Activation.hpp"
#include "Losses/Loss.hpp"

class Network{
    private:
        std::vector<std::unique_ptr<Layer>> layers;
        std::vector<std::unique_ptr<Activation>> activations;
        std::unique_ptr<Loss> loss;
    public:
        void addLayerAndActivation(std::unique_ptr<Layer> layer, std::unique_ptr<Activation> activation);
        void setLoss(std::unique_ptr<Loss> loss);
        Eigen::VectorXd forwardPass(const Eigen::VectorXd& input);
        double backwardPass(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target);

        void printWeightsAdjoints();
        void updateWeights();

        
};

#endif