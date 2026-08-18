#!/bin/bash

g++ -I/C:/msys64/mingw64/include/eigen3 Generate_independent_Data.cpp -o Executable_Gen_Independent_Data
g++ -I/C:/msys64/mingw64/include/eigen3 Independent_Grad_Descent.cpp Functions.cpp Read_File.cpp -o Executable_Independent_Grad_Descent
g++ -I/C:/msys64/mingw64/include/eigen3 Inexact_MCGD.cpp Functions.cpp Read_File.cpp Grad_Approx.cpp -o Executable_Inexact_Grad_Descent
