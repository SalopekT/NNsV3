#include "Network.hpp"

void Network::addLayerAndActivation(std::unique_ptr<Layer> layer, std::unique_ptr<Activation> activation){
    this->layers.push_back(std::move(layer));
    this->activations.push_back(std::move(activation));
}

void Network::setLoss(std::unique_ptr<Loss> loss){
    this->loss = std::move(loss);
}

Eigen::VectorXd Network::forwardPass(const Eigen::VectorXd& input){
    Eigen::VectorXd helper = input;
    for (int i=0;i<this->layers.size();i++){
        Eigen::VectorXd latent = layers.at(i)->simpleCalculateOutput(helper);
        Eigen::VectorXd latentAct = activations.at(i)->calculate(latent);
        helper = latentAct;
    }
    return helper;
}