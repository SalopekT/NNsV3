#ifndef LOSS
#define LOSS

#include <Eigen/Dense>
#include <iostream>

class Loss{
    protected:
        int dimensionInput;
        Eigen::VectorXd adjoint;
        Eigen::VectorXd input;
    public:
        virtual double calculate(const Eigen::VectorXd& input, const Eigen::VectorXd& target) = 0;
        virtual Eigen::VectorXd getAdjoint(const Eigen::VectorXd& input, const Eigen::VectorXd& target) = 0;
};


#endif