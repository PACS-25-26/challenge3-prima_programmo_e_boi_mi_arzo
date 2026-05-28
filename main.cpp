
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <numbers>
#include <omp.h>

#include "include/Parameters.hpp"
#include "include/Laplacian.hpp"
#include "include/PrintVar.hpp"
#include "include/Data.hpp"

int main(){

    // --- Parameters definition ---

    // // Read the parameters from json
    // const std::string FileName = "parameters.json";
    // Parameters p(FileName);

    // Spostiamo questi nel json e li decidiamo a runtime ?
    // Constexpr parameter for template class creation <--- ??? std::array per uno studio di convergenza? ???
    constexpr Test::Point bl = Test::bottom_left;
    constexpr Test::Point tr = Test::top_right;
    constexpr int Nx         = Test::Nx; // Nx = number of subinterval in direction x => Nx +1 number of nodes in direction x
    constexpr int Ny         = Test::Ny; // Ny = number of subinterval in direction y => Ny +1 number of nodes in direction x
    constexpr double hx      = Test::hx;
    constexpr double hy      = Test::hy;

    // Check Nx = Ny --> in the exercise a square grid is requested
    if constexpr(Nx != Ny)
        std::cerr<<"A square grid is requested, Nx must be equal to Ny"<<std::endl;

    // Compute h (same for x and y as specified in the text)
    const double h = 1 / static_cast<double>(Nx);

    // --- Define and apply f to the grid ---

    // Define lambda function f

    // auto lambda_f = [pi = std::numbers::pi](double x, double y){ 
    //     return 8 * (pi*pi) * std::sin(2*pi*x) * std::sin(2*pi*y);
    // };

    // Initialize data
    Test::ForcingTerm       force;
    Test::BoundaryCondition boundary_condition;
    Test::ExactSolution     exact_sol;

    // Initialize grid data
    Eigen::Array<double, Nx+1, Ny+1> f;
    Eigen::Array<double, Nx+1, Ny+1> u_ex;
    Eigen::Array<double, Nx+1,    1> bcTop; // <--- potremmo prenderle dalla soluzione esatta
    Eigen::Array<double, Nx+1,    1> bcBottom;
    Eigen::Array<double,    1, Ny+1> bcLeft;
    Eigen::Array<double,    1, Ny+1> bcRight;
    
    // --- Fill grid data ---
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=0; i<(Nx+1); ++i){
        for(int j=0; j<(Ny+1); ++j){
            f(i,j) = force(0.0 + i*h , 0.0 + j*h);
            u_ex(i,j) = exact_sol(0.0 + i*h , 0.0 + j*h);
        }
    }

    // Fill boundary conditions top and bottom
    #pragma omp parallel for
    for(int i=0; i<(Nx+1); ++i){
        bcTop(i)    = boundary_condition(bl[0] + i*h, tr[1]);
        bcBottom(i) = boundary_condition(bl[0] + i*h, bl[1]);
    }

    // Fill boundary conditions left and right
    #pragma omp parallel for
    for(int j=0; j<(Ny+1); ++j){
        bcLeft(j)  = boundary_condition(bl[0], bl[1] + j*h);
        bcRight(j) = boundary_condition(tr[0], bl[1] + j*h);
    }


    // Check correctness
    print_var("f", f);
    print_var("bcTop", bcTop);
    print_var("bcBottom", bcBottom);
    print_var("bcLeft", bcLeft);
    print_var("bcRight", bcRight);
    print_var("u_ex", u_ex);

    // // --- Initialize the solver ---
    // Operator::Laplacian<Nx, Ny> laplacian( 
    //                                 /*bcTop =*/     Eigen::Array<double,    1, Nx+1>::Constant(p.bcTop),
    //                                 /*bcBottom =*/  Eigen::Array<double,    1, Nx+1>::Constant(p.bcBottom),
    //                                 /*bcLeft =*/    Eigen::Array<double, Ny+1,    1>::Constant(p.bcLeft),
    //                                 /*bcRight =*/   Eigen::Array<double, Ny+1,     1>::Constant(p.bcRight),
    //                                 /*f = */ f,
    //                                 /*h =*/ h
    //                             );
    
    return 0;
}



