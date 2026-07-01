#ifndef LOSS
#define LOSS

#include <Eigen/Dense>
#include <iostream>

class Loss{
    protected:
        int dimensionInput;
        Eigen::VectorXd adjoint;
};


#endif