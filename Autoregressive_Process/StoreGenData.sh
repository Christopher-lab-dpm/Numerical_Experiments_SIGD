#!/bin/bash

# Create an independent folder for data storage --> Override existing files there if same name
# Idea --> We plan to re-use this data later on

mkdir -p IndependentDataStorage/

cp A_mat.txt u_vec.txt data_eps.txt IndependentDataStorage/