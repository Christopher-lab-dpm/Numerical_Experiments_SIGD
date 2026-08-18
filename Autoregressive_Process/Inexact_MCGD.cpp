#include "Functions.h"
#include "Read_File.h"
#include "Grad_Approx.h"
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <fstream> // file stream
#include <random>
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

    // Number of training steps used
    const int epoch_num = std::stoi(argv[3]);

    // Initialize the starting parameter of the algorithm
    double eps_k = std::stod(argv[4]);
    const double theta = std::stod(argv[5]);
    const double mu = std::stod(argv[6]); // Needs to be greater than 2 for certain properties to hold.
    const double step_exponent = 0.501; // Use this exponent when determining step size

    // First --> Read in the data tuple {eps2_list, eps1_list}
    std::tuple< std::vector<int>, std::vector<VectorXd> > tuple_eps = read_data_file(d , sample_size);

    // Access the individual elements of the tuple 
    std::vector<int> eps2_list = std::get<0>(tuple_eps);
    std::vector<VectorXd> eps1_list = std::get<1>(tuple_eps);

    /**
     * Inexact Markov Chain Gradient Descent with convex lofistic loss --> Trains using Markov Draws
     */

    // Initilize the parameter we will learn
    Eigen::VectorXd x_k = VectorXd::Zero(d);

    /**
     * Initialize the initial parameter of the Markov Chain
     */
    VectorXd eps_1 = VectorXd::Zero(d);
    int eps_2 = 0; 

    /**
     * Create lists to save training info which we will save to a file later
     */

    // List to save the value of x_k at each iteration
    std::vector<VectorXd> x_list;
    x_list.reserve(epoch_num);
    // List to save the losses at each iteration
    std::vector<double> loss_over_time;
    loss_over_time.reserve(epoch_num);
    // List which tracks the distance used in the forward finite difference method
    // at each successful iteration/update of the algorithm
    std::vector<double> delta_list;
    delta_list.reserve(epoch_num);
    // List use to track the value of eps_k at each iteration
    // This will allow us to know the allowable upper error bound on our approxmiation of the gradient
    std::vector<double> eps_k_list;
    eps_k_list.reserve(epoch_num);
    // List use to track the value of i_k at each iteration
    // This will allow us to know the value of i on the each successful update of the algorithm
    std::vector<int> i_k_list;
    i_k_list.reserve(epoch_num);
    // List which saves the approximate gradient g_k on each succesfull interation of the algorithm
    std::vector<VectorXd> g_k_list;
    g_k_list.reserve(epoch_num);
    // Track the norm f the difference between approximate gradient and the true gradient 
    std::vector<double> deriv_norm_diff_list;
    deriv_norm_diff_list.reserve(epoch_num);
    // Track the net sum of the error terms of the between the approximate and true gradient
    std::vector<double> ei_sum_list;
    ei_sum_list.reserve(epoch_num);


    /*
    * Use an anonymous function here (and call it) to initialize constant variable.
    * In this case, we are initializing the first standard basis vector of R^d
    */
    const VectorXd e1 = [d]{VectorXd v = VectorXd::Zero(d);
                            v(0) = 1.0;
                            return v;
                            }();

    // Read in the "u" vector used to generate the data
    VectorXd u = read_u_file(d);
    MatrixXd A = read_matrix_file(d);

    // Set up the random number generator for replicability
    std::mt19937 rng(446); // fixed seed for the random number generator
    std::normal_distribution<double> std_normal(0.0, 1.0);
    std::bernoulli_distribution keep_true(0.8); // keep true label with prob 0.8


    /**
     * Begin training over convex loss function using samples drawn from the Markov Chain
     */

    for(int k=1; k <= epoch_num; k++){
        /**
         * Generate the data in the Markov Chain at step k
         */

        // Get the value of eps_1 as A * eps1 + e1 * N(0,1)
        eps_1 = A * eps_1 + e1 * std_normal(rng);
        // Get the deterministic label for eps2
        int eps2_det = (u.dot(eps_1) > 0.0)? 1:0;
        // With probability 0.8, keep the current label, else it flips.
        bool keep = keep_true(rng);
        eps_2 = keep ? eps2_det : (1-eps2_det);

        /***** Perform Inexact gradient step ******/
        int i = 0; 
        VectorXd g_k = VectorXd::Zero(d);
        double diff_norm = 0.0;
        double ei_sum = 0.0;
        double stepsize = 1/std::pow(static_cast<double>(k), step_exponent);
        double L_lip = 2.0;
        double gradient_error = std::min(std::pow(theta,i)*eps_k, stepsize*std::pow(std::log(k),2));
        double distance = 0.99 * (2.0 * gradient_error)/(L_lip * std::pow(x_k.size(),0.5));

        // Avoid logical issues on the first run of the loop 
        if(k==1){
            gradient_error = std::pow(theta,i)*eps_k;
            distance = 0.99 * (2.0 * gradient_error)/(L_lip * std::pow(x_k.size(),0.5));
        }

        // Run while loop to find, i_k, g_k, and delta_k such that the inequalities of SIGD are satisfied
        while(true){
            if(i >= 50 ){
                // Break since the algorithm got stuck and the gradient is essentially zero
                break;
            }
            // Approximate Gradient via Forward Finite Difference methods
            g_k = finite_forward_diff_approx(x_k, eps_1, eps_2, distance);
            // Get the norm of the difference between the approximate gradient and the true gradient
            diff_norm = (g_k - convex_logistic_loss_gradient(x_k, eps_1,eps_2)).norm();
            // Get the net sum of the errors
            ei_sum = (g_k - convex_logistic_loss_gradient(x_k, eps_1,eps_2)).norm();
            
            // Check the conditions of the SIGD Algorithm 
            // We don't check error bound on gradient error since it is theoreticaly guranteed to hold
            if(g_k.norm() > std::pow(theta,i)*eps_k*mu){ 
                // Update the parameter
                x_k = x_k - stepsize * g_k;
                // Save the value of eps_k used at this time step
                eps_k_list.push_back(eps_k);
                // Update eps_k for the next time step
                eps_k = std::pow(theta,i) * eps_k;
                break;
            } else{
                // Increment i since our gradient didn't workout for the current i
                ++i;
                // Re-calculate the allowed gradient error upper bound
                gradient_error = std::min(std::pow(theta,i)*eps_k, stepsize*std::pow(std::log(k),2));
                // Re-calculate the distance to be used in the forward finite difference method based on 
                // the new gradient error
                distance = 0.99 * (2.0 * gradient_error)/(L_lip * std::pow(x_k.size(),0.5));
            
                // Avoid logical issues on the first run of the loop 
                if(k==1){
                    gradient_error = std::pow(theta,i)*eps_k;
                    distance = 0.99 * (2.0 * gradient_error)/(L_lip * std::pow(x_k.size(),0.5));
                }

            }



        }


        // Save the parameters we are tracking from the current iteration of the for loop
        x_list.push_back(x_k);
        delta_list.push_back(distance);
        i_k_list.push_back(i);
        g_k_list.push_back(g_k);
        deriv_norm_diff_list.push_back(diff_norm);
        ei_sum_list.push_back(diff_norm);

        // Calculate loss over the dataset after updating the parameter
        double L = 0.0;
        for(int j=0; j < sample_size; j++ ){
            L  = (convex_logistic_loss_func(x_k, eps1_list[j],  eps2_list[j])/static_cast<double>(sample_size)) + L;
        }
        // Save the loss from the current time step
        loss_over_time.push_back(L);

    }

    /**
     * Simulation is complete --> Now write out all the parameter saved during training
     */

    // Open the file stream. Override the file if it already exists
    std::ofstream out("INEXACT_MCGD_loss_training.txt");

    // Write out the trianing losses
    for(int i=0; i< loss_over_time.size();++i){
        out << std::setprecision(16) << loss_over_time[i];
        if(i != loss_over_time.size()-1) out << ","; 
    }
    out << "\n";
    out.close();
    std::cout << "Saved training losses \n";

    // Save the x_k training parameters 
    out.open("x_k_training.txt");
    // Write out the trianing losses
    for(int i=0; i< x_list.size();++i){
        const VectorXd& v = x_list[i];
        for(int j=0; j < v.size() ; ++j){
            out << std::setprecision(16) << v[j];
            if(j != v.size()-1) out << ","; 
        }
        out << "\n";
    }
    out.close();
    std::cout << "Saved x_k training params \n";

    // Save the values of delta used in the forward finite difference method
    out.open("delta_k.txt");
    // Write out the trianing losses
    for(int i=0; i< delta_list.size();++i){
        out << std::setprecision(16) << delta_list[i];
        if(i != delta_list.size()-1) out << ","; 
    }
    out << "\n";
    out.close();
    std::cout << "Saved deltas \n";

    // Save the values of eps_k used in the errors bounds at time k 
    out.open("epsilon_k.txt");
    for(int i=0; i< eps_k_list.size();++i){
        out << std::setprecision(16) << eps_k_list[i];
        if(i != eps_k_list.size()-1) out << ","; 
    }
    out << "\n";
    out.close();
    std::cout << "Saved epsilon k \n";

    // Save the value of i through at each training iteration k
    out.open("i_k.txt");
    for(int i=0; i< i_k_list.size();++i){
        out << i_k_list[i];
        if(i != i_k_list.size()-1) out << ","; 
    }
    out << "\n";
    out.close();
    std::cout << "Saved i_k \n";

    // Save the approximate gradients from training
    out.open("g_k_training.txt");
    for(int i=0; i< g_k_list.size();++i){
        const VectorXd& v = g_k_list[i];
        for(int j=0; j < v.size() ; ++j){
            out << std::setprecision(16) << v[j];
            if(j != v.size()-1) out << ","; 
        }
        out << "\n";
    }
    out.close();
    std::cout << "Saved g_k training approximate gradient \n";

    // Write out the norm of the diffence between the true and approximate gradient
    out.open("derive_diff.txt");
    for(int i=0; i< deriv_norm_diff_list.size();++i){
        out << std::setprecision(16) << deriv_norm_diff_list[i];
        if(i != deriv_norm_diff_list.size()-1) out << ","; 
    }
    out << "\n";
    out.close();
    std::cout << "Saved derive_diff \n";

    // Write out the values of the net sum of the errors of the difference in gradient at each iteration
    out.open("ei_sum.txt");
    for(int i=0; i< ei_sum_list.size();++i){
        out << std::setprecision(16) << ei_sum_list[i];
        if(i != ei_sum_list.size()-1) out << ","; 
    }
    out << "\n";
    out.close();
    std::cout << "Saved ei_sum \n";


}