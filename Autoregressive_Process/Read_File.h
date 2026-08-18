#include <vector>
#include <tuple>
#include <Eigen/Dense>

/**
 * Function to read in the data from data eps and returns the two lists
 */
std::tuple< std::vector<int>, std::vector<Eigen::VectorXd> > read_data_file(int eps1_size , int data_entries);

/**
 * Function which reads the text file "u_vec.txt" and returns the vector 'u' used in simulation
 */
Eigen::VectorXd read_u_file(int d);

/**
 * Function which reads the text file "A_mat.txt" and returns the matrix A
 */
Eigen::MatrixXd read_matrix_file(int d);

