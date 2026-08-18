#!/bin/bash

# If the data has already been generated, then run the independent and IMCGD algorithms


# Dimension of the vector "u"
dimension=$1 
# The number of independent samples which will be collected from the Markov Chain
sample_size=$2
# The number of steps which will be taken before sampling the Markov Chain for a single independent data point
chain_steps=$3
# The total number of iteration used for training in the any of the Descent Type Algorithms
epoch_num=$4

# Eps_k is the initial starting value of the error bound in the Inexact Markov Chain Gradient Descent Algorithm
eps_k=$5


# Generate the data which will be used for all the ablation studies
echo "Generating Independent Data"
./Executable_Gen_Independent_Data "$dimension" "$sample_size" "$chain_steps"
echo "Running Gradient Descent"
./Executable_Independent_Grad_Descent.exe "$dimension" "$sample_size" "$epoch_num"
# Save a copy of the independent generated data to a seperate folder (we will re-use it later)
./StoreGenData.sh
for mu in 2.5 2 1.5 1.25; do

    for theta in 0.9 0.75 0.5 0.25; do
        touch MetaData.txt

        echo "dimension = $dimension" >> MetaData.txt
        echo "sample_size = $sample_size" >> MetaData.txt
        echo "chain_steps = $chain_steps" >> MetaData.txt
        echo "epoch_num = $epoch_num" >> MetaData.txt
        echo "eps_k = $eps_k" >> MetaData.txt
        echo "theta = $theta" >> MetaData.txt
        echo "mu = $mu" >> MetaData.txt

        echo -e "\n Running ISGD simulation with mu=$mu and theta=$theta"

        ./Executable_Inexact_Grad_Descent.exe "$dimension" "$sample_size" "$epoch_num" "$eps_k" "$theta" "$mu"
        # Store all the generated and data used in a backup folder
        ./StoreSimData.sh
        # Put the independent data back in the current directory for the next run of the script
        cp -a IndependentDataStorage/* ./
    done

done