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

double Network::backwardPass(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target){
    double loss = this->loss->calculate(predicted, target);
    Eigen::VectorXd adjointPrev = this->loss->calculateAdjoint(target);
    for (int i=this->layers.size()-1;i>=0;i--){
        Eigen::VectorXd adjointAct = this->activations.at(i)->calculateAdjoint(adjointPrev);
        Eigen::MatrixXd adjointLinearWeights = this->layers.at(i)->calculateAdjointWeights(adjointAct);
        Eigen::VectorXd adjointLinearInput = this->layers.at(i)->calculateAdjointInput(adjointAct);
        adjointPrev = adjointLinearInput;
    }
    return loss;
}


void Network::printWeightsAdjoints(){
    for (int i=this->layers.size()-1;i>=0;i--){
        std::cout << "Layer: " << i << std::endl;
        std::cout << this->layers.at(i)->getAdjointWeights() << std::endl;
    }
}

void Network::updateWeights(){
    this->loss->resetAdjoint();
    for (int i=this->layers.size()-1;i>=0;i--){
        this->layers.at(i)->updateWeights();
        this->layers.at(i)->resetAdjointWeights();
        this->layers.at(i)->resetAdjointInput();
        this->activations.at(i)->resetAdjoint();
    }
}