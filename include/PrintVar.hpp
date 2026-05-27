#pragma once

#include <string>
#include <iostream>

template <typename T>
void print_var(const std::string& name, const T& var){
    std::cout << name << " = \n" << var << std::endl << std::endl;
}