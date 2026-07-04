#ifndef ACTIVATION
#define ACTIVATION

#include <Eigen/Dense>
#include <iostream>
class Activation{
    protected:
        int dimension;
        Eigen::VectorXd input;
        Eigen::VectorXd adjoint;


    public:
        Activation(int dimension) : dimension(dimension) {
            this->adjoint = Eigen::VectorXd::Zero(dimension);
        };
        virtual Eigen::VectorXd calculate(const Eigen::VectorXd& input) = 0;
        virtual Eigen::VectorXd calculateAdjoint(const Eigen::VectorXd& prevAdjoint) = 0;
        void resetAdjoint(){this->adjoint.setZero(dimension);};
        Eigen::VectorXd getAdjoint(){return this->adjoint;};
        virtual ~Activation() {};
};

#endif