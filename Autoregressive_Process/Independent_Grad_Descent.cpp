#include "Functions.h"
#include "Read_File.h"
#include <Eigen/Dense>
#include <vector>
#include <tuple>
#include <fstream> // file stream
#include <iomanip>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(int argc, char* argv[]){
    /**
     * Simulation Parameters
     */
    const int d = std::stoi(argv[1]);
    const int sample_size =  std::stoi(argv[2]);
    const int epoch_num =  std::stoi(argv[3]); // Number of training steps used in the gradient descent/MCGD
    const double step_exponent = 0.501; // Use this exponent when determining step size

    // First --> Read in the data tuple {eps2_list, eps1_list}
    std::tuple< std::vector<int>, std::vector<VectorXd> > tuple_eps = read_data_file(d , sample_size);

    // Access the individual elements of the tuple 
    std::vector<int> eps2_list = std::get<0>(tuple_eps);
    std::vector<VectorXd> eps1_list = std::get<1>(tuple_eps);


    /**
     * Gradient descent over convex logistic loss function --> Use the independent data
     */

    // Parameter vector which we will update during training
    VectorXd param = VectorXd::Zero(d);
    // List of losses which we will track and save during training
    std::vector<double> losses;
    losses.reserve(epoch_num);

    for(int epoch = 1; epoch <= epoch_num ; ++epoch){
        // Compute the average loss over the data set
        double L  = 0.0;
        for(int j=0; j < sample_size; j++ ){
            L  = (convex_logistic_loss_func(param, eps1_list[j],  eps2_list[j])/static_cast<double>(sample_size)) + L;
        }
        // Track the calculated loss
        losses.push_back(L);


        // Compute the average gradient over the dataset
        VectorXd grad = VectorXd::Zero(d);
        for (int j = 0; j < sample_size; j++) {
            grad = (convex_logistic_loss_gradient(param, eps1_list[j], eps2_list[j]) / static_cast<double>(sample_size)) + grad;
        }
        // stepsize = 1 / (epoch^0.501)
        double stepsize = 1.0 / std::pow(static_cast<double>(epoch), step_exponent); // used 0.501 previously but it stopped progress too fast
        // Update the parameter
        param =  param - (stepsize * grad);
    }

    // Save file of the training losses over the independet data in the current directory
    std::ofstream out("objective_loss_training.txt");

    // Write the losses to a file in chronological order
    for (int i = 0; i < losses.size(); i++) {
            out << std::setprecision(16) << losses[i];
            if( i != losses.size()-1) out << "," ;
        }
        out << "\n";
        
    out.close();
    std::cout << "Saved losses to objective_loss_training.txt \n";
}