
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
    Test::ExactSolution     exact_sol;
    Test::BoundaryCondition boundary_condition; // same as ExactSolution thanks to alias

  
    //______________________________________________________________________________________

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

    // Compute number of rows assigned to each rank balancing eccess
    unsigned quotient = nx / size;
    unsigned unmatchedRows = nx % size;
    unsigned local_nx = (rank < unmatchedRows) ? (quotient+1) : quotient; 

    //______________________________________________________________________________________

    // --- Initialize top and bottom boundary conditions for the local problem ---
    
    // For rank 0 and rank size --> use top and bottom, for the others use 0 as empty initialization
    Eigen::VectorXd local_bcTop = Eigen::VectorXd::Zero(local_nx+1); 
    Eigen::VectorXd local_bcBottom = Eigen::VectorXd::Zero(local_nx+1);
    
    // Overwrite the top and bottom boundaries
    if(rank == 0){
        #pragma omp parallel for
        for(unsigned i=0; i<(local_nx+1); ++i)
            local_bcTop(i) = boundary_condition(bl[0] + i*h, tr[1]);
    }
    else if(rank == size-1){
        #pragma omp parallel for
        for(unsigned i=0; i<(local_nx+1); ++i)
         local_bcBottom(i) = boundary_condition(bl[0] + i*h, bl[1]);
    }

    // --- Initialize left and right boundary conditions for the local problem ---

    // For all the rank initialize boundary condition on the left and right with 
    // the boundaries condition given in the exercise 
    Eigen::VectorXd local_bcLeft(ny+1);
    Eigen::VectorXd local_bcRight(ny+1);
    
    #pragma omp parallel for
    for(unsigned j=0; j<(ny+1); ++j){
        local_bcLeft(j)  = boundary_condition(bl[0], bl[1] + j*h);
        local_bcRight(j) = boundary_condition(tr[0], bl[1] + j*h);
    }

    //______________________________________________________________________________________

    // --- Initialize local grid data ---
    Eigen::ArrayXXd local_f{(local_nx+1)+2, ny+1};
    Eigen::ArrayXXd local_u_ex{(local_nx+1)+2, ny+1};

    // --- Initialize the local final solution matrix with ghost rows ---
    Eigen::ArrayXXd local_U0{Eigen::ArrayXXd::Zero((local_nx+1)+2 , ny+1)};
    Eigen::ArrayXXd local_U {Eigen::ArrayXXd::Zero((local_nx+1)+2, ny+1)};

    // --- Fill grid data ---
    // Compute the offset
    unsigned row_offset = 0;
    for(int r = 0; r < rank; ++r)
        row_offset += (r < unmatchedRows) ? (quotient+1) : quotient;

    // Update all the rows, also the upper ghost of the rank 0 (never used) 
    #pragma omp parallel for collapse(2) schedule(static)
    for(unsigned i = 0; i < (local_nx+1)+2; ++i){
        for(unsigned j = 0; j < ny+1; ++j){
            local_f(i, j)    = force(    bl[0] + (row_offset-1 + i)*h, bl[1] + j*h);
            local_u_ex(i, j) = exact_sol(bl[0] + (row_offset-1 + i)*h, bl[1] + j*h);
        }
    }

    //______________________________________________________________________________________

    // --- Initialize the solver ---
    Operator::Laplacian laplacian( 
        /*local_bcTop =*/     local_bcTop,
        /*local_bcBottom =*/  local_bcBottom,
        /*local_bcLeft =*/    local_bcLeft,
        /*local_bcRight =*/   local_bcRight,
        /*local_f = */ local_f,
        /*h = */ h,
        /*rank =*/ rank,
        /*size = */ size,
        /*nx = */ local_nx
    );

    // --- Solve ---
    int global_converged = false;
    bool local_converged = false;

    for(unsigned k=0; k < maxIt; ++k){

        // --- Send and receive necessary data of the common rows ---

        // Exchange with upper rank --> send my top row, in the upper ghost row
        if(rank < size-1)
            MPI_Sendrecv(
                local_U0.row(local_nx).data(),   ny+2, MPI_DOUBLE, rank+1, 0,
                local_U0.row(local_nx+1).data(), ny+2, MPI_DOUBLE, rank+1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

        // Exchange with lower rank --> send my bottom row, in the lower ghost row
        if(rank >0 )
            MPI_Sendrecv(
                local_U0.row(1).data(), ny+2, MPI_DOUBLE, rank-1, 0,
                local_U0.row(0).data(), ny+2, MPI_DOUBLE, rank-1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

        //  --- Apply the stencil to the local grid ---
        laplacian(local_U0, local_U);

        // --- Check global convergence ---

        // Check local convergence (su cosa va testata? tol comune o va divisa?)
        if(std::sqrt(h*(local_U - local_U0).square().sum()) < tol){
            local_converged = true;
        }

        // Check global convergence
        MPI_Allreduce(&local_converged, &global_converged, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        // Checks convergence
        if(global_converged)
            break;

        // Update step
        local_U0 = local_U;
    }

    // If convergence was not reached within maxIt print a message
    if(rank == 0){
        if(!global_converged){
            std::cout << "Jacobi iterations did not converge!" << std::endl;
            return 1;
        }
    }

    //______________________________________________________________________________________    

    // --- Compose all the solution matrix ---

    // Declare the empty variable for all the ranks
    Eigen::ArrayXXd global_U;

    // Prepare to receive global U
    std::vector<int> recvcounts(size, 0);
    std::vector<int> offsets(size, 0);

    if(rank == 0){
        // Resize it only for rank 0
        global_U.resize(nx+1, ny+1);

        int sum = 0;

        // Keep trace of how many elements each process has
        for(int i = 0; i < size; ++i) {
            int rows_i = (i < unmatchedRows) ? (quotient+1) : quotient;
            recvcounts[i] = (rows_i + 1) * (ny + 1);
            offsets[i] = sum;
            sum += recvcounts[i];
        }
    }    

    // Gatherv just the internal part of U
    MPI_Gatherv(local_U.data() + (ny+1), (local_nx+1)*(ny+1), MPI_DOUBLE, global_U.data(), 
                recvcounts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Display the solution 
    print_var("Global U", global_U);

    
    return 0;
}



