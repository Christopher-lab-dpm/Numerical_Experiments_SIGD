#include "Functions.h"
#include <Eigen/Dense>
#include <cmath>

using Eigen::VectorXd;

// Declare the sigmoid function --> Not vectorized
inline double sigmoid(double t) {
    return(1.0 / (1.0 + std::exp(-t)));
}

// Declare the convex logistic loss function
double convex_logistic_loss_func(const VectorXd& x, const VectorXd& eps1, int eps2){
    double z = x.dot(eps1);
    double s = sigmoid(z); 
    return (-1.0 * static_cast<double>(eps2) * std::log(s) - (1.0-static_cast<double>(eps2))*std::log(1.0-s));
}

/**
 * Returns the gradient of the convex_logistic_loss_func with respect to the vector x
 */
VectorXd convex_logistic_loss_gradient(const VectorXd& x, const VectorXd& eps1 , int eps2){
    double z = x.dot(eps1);
    double s = sigmoid(z);
    return (eps1 * (s - static_cast<double>(eps2)));
}

