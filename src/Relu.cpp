#include "Relu.hpp"

Eigen::VectorXd Relu::calculate(const Eigen::VectorXd& input){
    return input.array().max(0.0);
}