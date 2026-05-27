
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <numbers>
#include <omp.h>

#include "include/Parameters.hpp"
#include "include/Laplacian.hpp"
#include "include/PrintVar.hpp"

int main(){

    // --- Parameters definition ---

    // Read the parameters from json
    const std::string FileName = "parameters.json";
    Parameters p(FileName);

    // Constexpr parameter for template class creation <--- ??? std::array per uno studio di convergenza? ???
    constexpr int Nx = 11;
    constexpr int Ny = 11;

    // Check Nx = Ny --> in the exercise a square grid is requested
    if constexpr(Nx != Ny)
        std::cerr<<"A square grid is requested, Nx must be equal to Ny"<<std::endl;

    // Compute h (same for x and y as specified in the text)
    const double h = 1 / (Nx-1);

    // --- Define and apply f to the grid ---

    // Define lambda function f

    auto lambda_f = [pi = std::numbers::pi](double x, double y){ 
        return 8 * (pi*pi) * std::sin(2*pi*x) * std::sin(2*pi*y);
    };

    // Apply it to the grid
    Eigen::Array<double, Nx+1, Ny+1> f;

    #pragma omp parallel for
    for(int i=0; i<(Nx+1); ++i)
        for(int j=0; j<(Nx+1); ++j)
            f(i,j) = lambda_f(0.0 + i*h , 0.0 + j*h);

    print_var("f", f);

    // --- Initialize the solver ---
    Operator::Laplacian<Nx, Ny> laplacian( 
                                    /*bcTop =*/     Eigen::Array<double,    1, Nx+1>::Constant(p.bcTop),
                                    /*bcBottom =*/  Eigen::Array<double,    1, Nx+1>::Constant(p.bcBottom),
                                    /*bcLeft =*/    Eigen::Array<double, Ny+1,    1>::Constant(p.bcLeft),
                                    /*bcRight =*/   Eigen::Array<double, Ny+1,     1>::Constant(p.bcRight),
                                    /*f = */ f,
                                    /*h =*/ h
                                );
    
    return 0;
}



