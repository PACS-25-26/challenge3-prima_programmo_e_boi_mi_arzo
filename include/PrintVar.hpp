#pragma once

#include <string>
#include <iostream>

template <typename T>
void print_var(const std::string& name, const T& var){
    std::cout << name << " = " << var << std::endl << std::endl;
}

template <typename T>
void print_eigen(const std::string& name, const T& var){
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
