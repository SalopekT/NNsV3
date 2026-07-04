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
        Loss(int dimensionInput) : dimensionInput(dimensionInput){
            this->adjoint = Eigen::VectorXd::Zero(dimensionInput);
        }
        virtual double calculate(const Eigen::VectorXd& predicted, const Eigen::VectorXd& target) = 0;
        virtual Eigen::VectorXd calculateAdjoint(const Eigen::VectorXd& target) = 0; //input vector is also needed but it is 
                                                                                //stored in the object
        void resetAdjoint(){this->adjoint.setZero(dimensionInput);};
        Eigen::VectorXd getAdjoint(){return this->adjoint;};
};


#endif