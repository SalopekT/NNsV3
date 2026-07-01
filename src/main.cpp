#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include "Layer.hpp"
#include "LinearLayer.hpp"
int main() {
    std::cout << "Hello, Eigen!" << std::endl;

    Eigen::Matrix2d matrix;
    matrix << 1, 2,
              3, 4;

    std::cout << "Matrix:\n" << matrix << std::endl;

    std::unique_ptr<Layer> l1 = std::make_unique<LinearLayer>(10,20);
    l1->printWeights();
    return 0;
}