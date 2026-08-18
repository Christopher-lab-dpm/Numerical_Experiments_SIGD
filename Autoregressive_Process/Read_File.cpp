#include <Eigen/Dense>
#include <fstream> // file stream
#include <sstream> // string stream
#include <vector>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Function which reads the text file "A_mat.txt" and returns the matrix A
 */
MatrixXd read_matrix_file(int d){
    // Open file stream to read from the file
    std::ifstream read_in("A_mat.txt");
    MatrixXd A = MatrixXd::Zero(d,d);

    std::string number;
    for(int i=0; i < d;  ++i){
        for(int j=0; j < d; ++j){
            getline(read_in, number, ',');
            A(i,j) = std::stod(number);
        }    
    }
    read_in.close();
    return A;
}


/**
 * Function which reads the text file "u_vec.txt" and returns the vector 'u' used in simulation
 */
VectorXd read_u_file(int d){
    // Open file stream to read from the file
    std::ifstream read_in("u_vec.txt");
    VectorXd u = VectorXd::Zero(d);

    std::string number;
    for(int i=0; i < d;  ++i){
        getline(read_in, number, ',');
        u(i) = std::stod(number);
    }
    read_in.close();
    return u;
}


/**
 * Function to read in the data from data eps and returns the two lists
 */
std::tuple< std::vector<int>, std::vector<VectorXd> > read_data_file(int eps1_size , int data_entries){
std::vector<VectorXd> eps1_list;
// Pre-allocate memory for efficiency
eps1_list.reserve(data_entries);
std::vector<int> eps2_list;
// Pre-allocate memory for efficiency
eps2_list.reserve(data_entries);

// Open file stream to read from the file
std::ifstream read_in("data_eps.txt");
std::string numbers;
std::string num_token;

for(int line=0; line < data_entries; ++line){
    getline(read_in, numbers); // Reads until "/n" newline charcater is read

    VectorXd cur_eps1 = VectorXd::Zero(eps1_size);
    int cur_eps2 = 0;

    int j = 0;
    std::stringstream ss(numbers);
    while(std::getline(ss, num_token, ',')){
        if(num_token.empty() || num_token == "\n" ) break;

        if(j == 0){
            cur_eps2 = std::stod(num_token); 
        } else { 
            cur_eps1(j-1) = std::stod(num_token); 
        }
        ++j;
    }
    eps1_list.push_back(cur_eps1);
    eps2_list.push_back(cur_eps2);
}

read_in.close();
return {eps2_list , eps1_list};
}
