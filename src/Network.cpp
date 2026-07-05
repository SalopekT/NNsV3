#include "Network.hpp"
#include "Layer.hpp"
#include "Activations/Activation.hpp"
#include "Losses/Loss.hpp"
#include "LinearLayer.hpp"
#include "Activations/Relu.hpp"
#include "Activations/Softmax.hpp"
#include <fstream>
#include <random>

Network::Network(){

}



//for testing
Network::Network(const std::string& path){
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open weights file: " + path);
    }
    std::string line;
    int layerIndex = 0;
    while (std::getline(file, line)) {
        if (line.size() > 0 && line[0] == '#') {
            std::istringstream header(line.substr(1)); // skip '#'
            int numInputs, numOutputs;
            header >> numInputs >> numOutputs;
            if (header.fail()) {
                throw std::runtime_error("Invalid header: " + line);
            }

            Eigen::MatrixXd weights = Eigen::MatrixXd::Zero(numOutputs,numInputs);

            
            for (int i = 0; i < numOutputs; i++) {
                if (!std::getline(file, line)) {
                    throw std::runtime_error("Unexpected EOF while reading weights");
                }
                std::istringstream rowStream(line);
                for (int j = 0; j < numInputs; j++) {
                    double val;
                    rowStream >> val;
                    if (rowStream.fail()) {
                        throw std::runtime_error("Invalid weight value in line: " + line);
                    }
                    weights(i,j)=val;
                }
            }
            //!!!! this is hardcoded !!!!
            std::unique_ptr<Layer> layer = std::make_unique<LinearLayer>(numInputs-1,numOutputs,weights);
            layers.push_back(std::move(layer));
            layerIndex++;
        }
    }
    for (int i=0;i<layers.size();i++){
        int outputSize = layers.at(i)->getDimensionOutput();
        if (i!=layers.size()-1){
            std::unique_ptr<Activation> activation = std::make_unique<Relu>(outputSize);
            activations.push_back(std::move(activation));
        }
        else{
            std::unique_ptr<Activation> activation = std::make_unique<Softmax>(outputSize);
            activations.push_back(std::move(activation));
        }
    }


}


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
    //std::cout  << "Output: "<< helper << std::endl;
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

void Network::updateWeights(double learningRate){
    this->loss->resetAdjoint();
    for (int i=this->layers.size()-1;i>=0;i--){
        this->layers.at(i)->updateWeights(learningRate);
        this->layers.at(i)->resetAdjointWeights();
        this->layers.at(i)->resetAdjointInput();
        this->activations.at(i)->resetAdjoint();
    }
}


void Network::storeWeightsInFileSystem(const std::string& path){
    std::cout << "Storing..." << std::endl;
    std::ofstream weightFile(path);
    for (const std::unique_ptr<Layer>& layer : this->layers){
        weightFile << "# ";
        weightFile << layer->getDimensionInput()+1 << " " << layer->getDimensionOutput() << "\n";
        for (int i=0;i<layer->getDimensionOutput();i++){
            for (int j=0;j<layer->getDimensionInput()+1;j++){
                weightFile << layer->getWeight(i,j);
                if (j!=layer->getDimensionInput()-1) weightFile << " ";
            }
            weightFile << "\n";
        }

    }
    weightFile.close();
}

void Network::stochasticGradientDescent(double learningRate, int numEpochs,
                                        const std::vector<std::vector<uint8_t>>& trainImages, const std::vector<uint8_t>& trainLabels){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, trainImages.size()-1);
    int counter = 1;
    double avg_loss = 0;
    for (int i=0;i<numEpochs;i++){
        for (int j=0;j<50000;j++){
            int index = dist(gen);
            std::vector<uint8_t> sample = trainImages[index];
            uint8_t label = trainLabels[index];
            
            //creating input data compatible with the neural network
            Eigen::VectorXd input;
            std::vector<double> inputHelper;
            for (int row = 0; row < 28; row++) {
                for (int col = 0; col < 28; col++) {
                    inputHelper.push_back(sample[row * 28 + col]/(127.5)-1);   //this is pixel/255*2 - 1 so it is normalized to [-1,1]
                }

            }
            input = Eigen::Map<Eigen::VectorXd>(inputHelper.data(), inputHelper.size());
            Eigen::VectorXd realOutput = Eigen::VectorXd::Zero(10);
            for (int i=0;i<10;i++){
                if ((int)label == i){
                    realOutput(i)=1;
                }
            }
            Eigen::VectorXd prediction = this->forwardPass(input);
            double loss = this->backwardPass(prediction,realOutput);
            //std::cout << loss << std::endl;
            this->updateWeights(learningRate);

            avg_loss+=loss;
            counter++;
            if (counter==5000){
                std::cout << "Average loss: " << avg_loss/5000 << std::endl;
                counter = 1;
                avg_loss=0;
            }

        }
    }
}

void Network::storeWeightsCumulative(){
    this->loss->resetAdjoint();
    for (int i=this->layers.size()-1;i>=0;i--){
        this->layers.at(i)->storeWeightsCumulative();
        this->layers.at(i)->resetAdjointWeights();
        this->layers.at(i)->resetAdjointInput();
        this->activations.at(i)->resetAdjoint();
    }
}

void Network::updateWeightsBatch(double learningRate, int batchSize){
    this->loss->resetAdjoint();
    for (int i=this->layers.size()-1;i>=0;i--){
        this->layers.at(i)->updateWeightsBatch(learningRate, batchSize);
        this->layers.at(i)->resetCumulativeAdjointWeights();
        this->layers.at(i)->resetAdjointWeights();
        this->layers.at(i)->resetAdjointInput();
        this->activations.at(i)->resetAdjoint();
    }
}

void Network::miniBatchGradientDescent(double learningRate, int numEpochs, int batchSize,
                                        const std::vector<std::vector<uint8_t>>& trainImages, const std::vector<uint8_t>& trainLabels){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, trainImages.size()-1);
    int counter = 1;
    double avg_loss = 0;
    for (int i=0;i<numEpochs;i++){
        for (int j=0;j<2000;j++){
            std::vector<std::vector<uint8_t>> batch;
            for (int k=0;k<batchSize;k++){
                int index = dist(gen);
                std::vector<uint8_t> sample = trainImages[index];
                uint8_t label = trainLabels[index];
                 //creating input data compatible with the neural network
                Eigen::VectorXd input;
                std::vector<double> inputHelper;
                for (int row = 0; row < 28; row++) {
                    for (int col = 0; col < 28; col++) {
                        inputHelper.push_back(sample[row * 28 + col]/(127.5)-1);   //this is pixel/255*2 - 1 so it is normalized to [-1,1]
                    }

                }
                input = Eigen::Map<Eigen::VectorXd>(inputHelper.data(), inputHelper.size());
                Eigen::VectorXd realOutput = Eigen::VectorXd::Zero(10);
                for (int i=0;i<10;i++){
                    if ((int)label == i){
                        realOutput(i)=1;
                    }
                }
                Eigen::VectorXd prediction = this->forwardPass(input);
                double sampleLoss = this->backwardPass(prediction,realOutput);
                avg_loss+=sampleLoss;
                this->storeWeightsCumulative();
            }

            this->updateWeightsBatch(learningRate,batchSize);
            counter++;
            if (counter==200){
                std::cout << "Average loss: " << avg_loss/(200*batchSize) << std::endl;
                counter = 1;
                avg_loss=0;
            }
            
        }
    }


}