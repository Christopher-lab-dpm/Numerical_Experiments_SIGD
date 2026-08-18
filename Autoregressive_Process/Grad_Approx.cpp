#include "Functions.h" 
#include <Eigen/Dense>

using Eigen::VectorXd;

/**
 * Approximate the gradient of the logistic loss function via the
 * finite forward difference method with respect to the parameter x and
 * using the indicated distance.
 * eps1 is the data and eps2 is the label
 * Reference: https://www.sfu.ca/sasdoc/sashtml/iml/chap11/sect8.htm
 */
VectorXd finite_forward_diff_approx(const VectorXd& x, const VectorXd& eps1, int eps2, double distance){
    VectorXd approx_grad(x.size());

    for(int i=0; i < x.size(); i++ ){
        VectorXd delta_x = x;
        delta_x(i) =  delta_x(i) + distance;

        approx_grad(i) = (convex_logistic_loss_func(delta_x, eps1, eps2) - convex_logistic_loss_func(x, eps1, eps2))/distance;
    }
    return(approx_grad);
}