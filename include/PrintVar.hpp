#pragma once

#include <string>
#include <iostream>
//#include <mpi.h>

template <typename T>
void print_var(const std::string& name, const T& var){
    std::cout << name << " = \n" << var << std::endl << std::endl;
}

void print_line(){
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "________________________________________________________________________________________________________" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

}

void print_debug(){
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "*******************************************************************************************************" << std::endl;
    std::cout << std::endl;
    std::cout << std::endl;

}

// template <typename T>
// void MPI_Print(const std::string& name, const T& var){
// }