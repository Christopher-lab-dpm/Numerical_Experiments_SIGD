#include <Eigen/Dense>

// Declare the sigmoid function --> Not vectorized
double sigmoid(double t);

/**
 * Declare the convex logistic loss function which is parametrized by x
 * and eps1 is the data and eps2 is the binary label
 */
double convex_logistic_loss_func(const Eigen::VectorXd& x, const Eigen::VectorXd& eps1, int eps2);

/**
 * Returns the gradient of the convex_logistic_loss_func with respect to the vector x
 * and where eps1 is the data vector and eps2 is the binary label
 */
Eigen::VectorXd convex_logistic_loss_gradient(const Eigen::VectorXd& x, const Eigen::VectorXd& eps1 , int eps2);
