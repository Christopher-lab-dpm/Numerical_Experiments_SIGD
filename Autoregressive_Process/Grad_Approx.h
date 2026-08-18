#include <Eigen/Dense>

Eigen::VectorXd finite_forward_diff_approx(const Eigen::VectorXd& x, const Eigen::VectorXd& eps1, int eps2, double distance);
    