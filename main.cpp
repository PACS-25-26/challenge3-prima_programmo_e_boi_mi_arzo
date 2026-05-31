
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <numbers>
#include <omp.h>
#include <mpi.h>

#include "include/Parameters.hpp"
#include "include/Laplacian.hpp"
#include "include/PrintVar.hpp"
#include "include/Data.hpp"
#include "include/Jacobi.hpp"

int main(int argc, char** argv){
    // --- Initialization for MPI ---
    MPI_Init(&argc, &argv);

    // Get rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Declare common variables
    unsigned nx, ny, maxIt;
    double h, tol;

    // Declare boundary points and common functions
    constexpr Test::Point bl = Test::bottom_left;
    constexpr Test::Point tr = Test::top_right;
    Test::ForcingTerm       force;
    Test::BoundaryCondition boundary_condition;
    Test::ExactSolution     exact_sol;

    // --- Parameters reading made by rank 0---
    if (rank == 0){
        // Read the parameters from json
        const std::string FileName = "parameters.json";
        Parameters p(FileName);

        // Compute h (same for x and y as specified in the text)
        h = 1 / static_cast<double>(p.N);

        // Compute the number of subdivisions in the two directions accordingly to h
        nx = static_cast<unsigned>((tr[0] - bl[0])/h);
        ny = static_cast<unsigned>((tr[1] - bl[1])/h);

        // Assign max number of iterations for the Broadcast
        maxIt = p.maxIt;

        // Assign tollerance for the Broadcast
        tol = p.tol;
    }

    // --- Rank 0 Broadcasts parameters to all the ranks ---
    MPI_Bcast(&nx, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxIt, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // If you want we can rebuild the parameter object in local????

    // Compute number of 
   

    // Initialize grid data
    Eigen::ArrayXXd    f(nx+1, ny+1);
    Eigen::ArrayXXd    u_ex(nx+1, ny+1);
    Eigen::RowVectorXd bcTop(nx+1); // <--- potremmo prenderle dalla soluzione esatta
    Eigen::RowVectorXd bcBottom(nx+1);
    Eigen::VectorXd    bcLeft(ny+1);
    Eigen::VectorXd    bcRight(ny+1);
    
    // --- Fill grid data ---
    #pragma omp parallel for collapse(2) schedule(static)
    for(unsigned i=0; i<(nx+1); ++i){
        for(unsigned j=0; j<(ny+1); ++j){
            f(i,j) = force(0.0 + i*h, 0.0 + j*h);
            u_ex(i,j) = exact_sol(0.0 + i*h , 0.0 + j*h);
        }
    }

    // --- Fill boundary conditions top and bottom ---
    #pragma omp parallel for
    for(unsigned i=0; i<(nx+1); ++i){
        bcTop(i)    = boundary_condition(bl[0] + i*h, tr[1]);
        bcBottom(i) = boundary_condition(bl[0] + i*h, bl[1]);
    }

    // --- Fill boundary conditions left and right ---
    #pragma omp parallel for
    for(unsigned j=0; j<(ny+1); ++j){
        bcLeft(j)  = boundary_condition(bl[0], bl[1] + j*h);
        bcRight(j) = boundary_condition(tr[0], bl[1] + j*h);
    }

    // Check correctness
    /*
    print_var("f", f);
    print_var("bcTop", bcTop);
    print_var("bcBottom", bcBottom);
    print_var("bcLeft", bcLeft);
    print_var("bcRight", bcRight);
    print_var("u_ex", u_ex);
    */

    // --- Initialize the solver ---
    Operator::Laplacian laplacian( 
        /*bcTop =*/     bcTop,
        /*bcBottom =*/  bcBottom,
        /*bcLeft =*/    bcLeft,
        /*bcRight =*/   bcRight,
        /*f = */ f,
        /*h = */ h
    );

    // --- Initialize the solution matrix ---
    Eigen::ArrayXXd U0{Eigen::ArrayXXd::Zero(nx+1, ny+1)};
    Eigen::ArrayXXd U{Eigen::ArrayXXd::Zero(nx+1, ny+1)};




    // --- Solve ---
    bool converged = false;
    for(unsigned k=0; k < maxIt; ++k){
        // Apply the stencil 
        laplacian(U0, U);

        // Check convergence
        if(std::sqrt(h*(U - U0).square().sum()) < tol){
            converged = true;
            break;
        }

        // Update step
        U0 = U;
    }

    // If convergence was not reached within maxIt print a message
    if(!converged){
        std::cout << "Jacobi iterations did not converge!" << std::endl;
        return 1;
    }
    
    print_var("U", U);

    
    return 0;
}



