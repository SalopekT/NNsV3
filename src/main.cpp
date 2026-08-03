#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <random>
#include "Layers/Layer.hpp"
#include "Layers/LinearLayer.hpp"
#include "Activations/Activation.hpp"
#include "Activations/Relu.hpp"
#include "Activations/Softmax.hpp"
#include "Network.hpp"
#include "Losses/CrossEntropyLoss.hpp"
#include "MNISTdata/FileReader.hpp"
#include "Layers/ConvolutionalLayer.hpp"
#include "Layers/MCConvolutionalLayer.hpp"
#include <vector>

int main() {
    /*std::unique_ptr<ConvolutionalLayer> convLayer1 = std::make_unique<ConvolutionalLayer>(9,3);
    convLayer1->printWeights();
    convLayer1->printConvMatrix();
    std::cout << "----------\n";
    Eigen::VectorXd input = Eigen::VectorXd::Random(9);
    std::cout << input << std::endl;
    std::cout << "----------\n";*/
    /*Eigen::VectorXd output = convLayer1->simpleCalculateOutput(input);
    std::cout << output << std::endl;*/
    //std::cout << "----------\n";

    //std::unique_ptr<Layer> mcConvLayer1 = std::make_unique<MCConvolutionalLayer>(1,32,9,3);
    //Eigen::VectorXd outputFirst = mcConvLayer1->simpleCalculateOutput(input);

    FileReader reader("C:\\Users\\tinsa\\Projects\\NNs\\train-images-idx3-ubyte\\train-images.idx3-ubyte", "C:\\Users\\tinsa\\Projects\\NNs\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");
    reader.read_mnist();

    auto& imgs = reader.get_images();
    auto& lbls = reader.get_labels();
    std::cout << "Images: " << imgs.size() 
            << " Labels: " << lbls.size() << "\n";

    size_t N = std::min<size_t>(30, imgs.size());

    std::vector<std::vector<uint8_t>> imgs_small(
        imgs.begin(),
        imgs.begin() + N
    );

    std::vector<uint8_t> lbls_small(
        lbls.begin(),
        lbls.begin() + N
    );

    std::unique_ptr<Layer> mcConvLayer1 = std::make_unique<MCConvolutionalLayer>(1,32,784,3);
    std::unique_ptr<Activation> mca1 = std::make_unique<Relu>(32*784);

    std::unique_ptr<Layer> l1 = std::make_unique<LinearLayer>(32*784,10);
    //l1->printWeights();
    std::unique_ptr<Activation> a1 = std::make_unique<Softmax>(10);

    std::unique_ptr<Layer> l2 = std::make_unique<LinearLayer>(256,128);
    //l2->printWeights();
    std::unique_ptr<Activation> a2 = std::make_unique<Relu>(128);

    std::unique_ptr<Layer> l3= std::make_unique<LinearLayer>(128,10);
    //l2->printWeights();
    std::unique_ptr<Activation> a3 = std::make_unique<Softmax>(10);
    std::unique_ptr<Loss> loss = std::make_unique<CrossEntropyLoss>(10);

    Network* net = new Network();
    net->addLayerAndActivation(std::move(mcConvLayer1),std::move(mca1));
    net->addLayerAndActivation(std::move(l1),std::move(a1));
    //net->addLayerAndActivation(std::move(l2),std::move(a2));
    //net->addLayerAndActivation(std::move(l3),std::move(a3));
    net->setLoss(std::move(loss));
    std::cout << imgs[0].size() << std::endl;
    std::cout << static_cast<int>(lbls[2]) << '\n';

    net->miniBatchGradientDescent(0.005,25,1,imgs,lbls);
    net->storeWeightsInFileSystem("weights8.txt");
    delete net;


    //testing
    /*Network* network = new Network("C:\\Users\\tinsa\\Projects\\NNsV3\\build\\Debug\\weights7.txt");
    FileReader reader("C:\\Users\\tinsa\\Projects\\NNs\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte", "C:\\Users\\tinsa\\Projects\\NNs\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte");
    reader.read_mnist();

    auto& imgs = reader.get_images();
    auto& lbls = reader.get_labels();
    std::cout << "Images: " << imgs.size() 
            << " Labels: " << lbls.size() << "\n";

    int successes = 0;
    int total = 0;
    std::cout << "Here\n";
    for (int j=0;j<imgs.size();j++){
        Eigen::VectorXd vec(imgs.at(j).size());
        for (size_t i = 0; i < imgs.at(i).size(); ++i) {
            vec(i) = static_cast<double>(imgs.at(j)[i]) / 127.5 - 1;
        }
        Eigen::VectorXd output = network->forwardPass(vec);
        Eigen::Index maxIndex;
        output.maxCoeff(&maxIndex);

        if (maxIndex == lbls[j]){
        successes++;
        }
        else{
            std::cout << static_cast<int>(lbls[j]) << std::endl;

        }
        total++;
        
    }

    std::cout << "Accuracy: " << (double) successes/total << "\n";
    
    
    std::cout << "\n";
    delete network;*/

    return 0;
}