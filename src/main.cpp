#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include "Layer.hpp"
#include "LinearLayer.hpp"
#include "Activations/Activation.hpp"
#include "Activations/Relu.hpp"
#include "Activations/Softmax.hpp"
#include "Network.hpp"
#include "Losses/CrossEntropyLoss.hpp"
#include "MNISTdata/FileReader.hpp"

int main() {
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


    
    std::unique_ptr<Layer> l1 = std::make_unique<LinearLayer>(784,64);
    l1->printWeights();
    std::unique_ptr<Activation> a1 = std::make_unique<Relu>(64);

    std::unique_ptr<Layer> l2= std::make_unique<LinearLayer>(64,10);
    l2->printWeights();
    std::unique_ptr<Activation> a2 = std::make_unique<Softmax>(10);

    std::unique_ptr<Loss> loss = std::make_unique<CrossEntropyLoss>(10);

    Network* net = new Network();
    net->addLayerAndActivation(std::move(l1),std::move(a1));
    net->addLayerAndActivation(std::move(l2),std::move(a2));
    net->setLoss(std::move(loss));
    std::cout << imgs[0].size() << std::endl;
    std::cout << static_cast<int>(lbls[0]) << '\n';
    delete net;

    return 0;
}