#include <Eigen/Dense>
#include <random>
#include <fstream> // file stream
#include <vector>
#include <iomanip>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;



int main(int argc, char* argv[]){
    /**
     * Simulation Parameters
     */
    const int d = std::stoi(argv[1]);
    const int sample_size = std::stoi(argv[2]);
    const int chain_steps = std::stoi(argv[3]); // Number of steps to take before we sample a Markov Chain


    /**
     * Declare data structure to store the drawn samples (eps1, eps2)
     */
    std::vector<VectorXd> eps1_list;
    // Pre-allocate memory for efficiency
    eps1_list.reserve(sample_size);
    std::vector<int> eps2_list;
    // Pre-allocate memory for efficiency
    eps2_list.reserve(sample_size);


    // Set up the random number generator for replicability
    std::mt19937 rng(345); // fixed seed for the random number generator
    // Set up the distributions used in the simulation
    std::uniform_real_distribution<double> unif_subdiag(0.8, 0.99);
    std::normal_distribution<double> std_normal(0.0, 1.0);
    std::bernoulli_distribution keep_true(0.8); // keep true label with prob 0.8

    // Create the sub-diagonal matrix with entires initialize from
    // a uniform distribution ~ U[0.8, 0.99]
    MatrixXd A = MatrixXd::Zero(d,d); // Declare the matrix
    for (int i =1 ; i<d ; i++){ // Start at index one (not zero) to proceed along the sub-diagonal
        A(i,i-1) = unif_subdiag(rng);
    }


    /*
    * Generate the parameter vector 'u' we wish to recover.
    * Initialize u from N(0,I) and scale such that ||u||_2 = 5
    */
    VectorXd u(d);
    for(int i=0; i < d; i++){
        u(i) = std_normal(rng);
    }
    // Scale u
    u = (u / u.norm()) * 5.0;


    /*
    * Use an anonymous function here (and call it) to initialize constant variable.
    * In this case, we are initializing the first standard basis vector of R^d
    */
    const VectorXd e1 = [d]{
                           VectorXd v = VectorXd::Zero(d);
                           v(0) = 1.0;
                           return v;
                         }();               



    /**
     * Create the independent dataset of the stationary distribution of size = sample_size 
     * by independently runnning a new markov chain for number of steps = chain_steps
     * and only save the last values of (eps1, eps2)
     */
    for(int n=0; n < sample_size ; ++n){
        // Initialize the starting point of the Markov Chain
        VectorXd eps1 = VectorXd::Zero(d);
        int eps2 = 0;
        for(int t=0; t < chain_steps; ++t){
            // Determine the next value of the data from the Markov Chain
            eps1 = A * eps1 + e1 * std_normal(rng);

            // Get the value of eps2 before it flips
            int eps2_det = (u.dot(eps1) > 0.0) ? 1 : 0;
            bool keep = keep_true(rng);
            // Noisy flip the value of eps2 with probability 0.8 
            eps2 = keep ? eps2_det : (1-eps2_det);
        }
        // Save the values from the end of walk along the Markov Chain
        eps1_list.emplace_back(std::move(eps1));
        eps2_list.emplace_back(eps2);
    }


    // ######### Now save the data used for the simulations to text files ########

    // Open output stream the data file we wish to write to
    std::ofstream out("data_eps.txt"); // Note: We are overwriting any existing file with this

    /*
    * Write to the file "data_eps.txt" where each row of the text file corresponds to
    * the last saved values of one full run of a Markov chain.
    * We save the data as: (eps2, eps1) respectively
    */
    for(int i=0; i < sample_size; ++i){
        out << eps2_list[i] << ",";
        out.flush();
        VectorXd& v = eps1_list[i];
        for (int j = 0; j < v.size(); ++j) {
            if(j < (v.size()-1)){
                out << std::setprecision(16) << v[j] << ",";
                out.flush();
            } else {
                out << std::setprecision(16) << v[j] << std::endl;
            }
        }
    }
    out.close();
    std::cout << "Saved samples to data_eps.txt\n";

    // Save the matrix we used in the Markov Chain    
    out.open("A_mat.txt");
    for (int i=0; i < d; ++i) {
        for(int j=0; j < d; ++j){
            // if A(i, j) != 0 continue --> maybe include this
            out << A(i, j) << ",";
            out.flush();
        }
    }
    out.close();
    std::cout << "Saved A matrix to A_mat.txt\n";
    
    // Write out the vector 'u' used for the simulation
    out.open("u_vec.txt");
    for (int i = 0; i < d; ++i) {
        out << u(i) << ",";
        out.flush();
    }
    out.close();
    std::cout << "Saved u to u_vec.txt\n";

}






