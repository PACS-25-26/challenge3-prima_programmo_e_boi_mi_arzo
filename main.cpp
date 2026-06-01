
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

    //______________________________________________________________________________________
   
    // Compute number of rows assigned to each rank balancing eccess
    unsigned quotient = nx / size;
    unsigned unmatchedRows = nx % size;
    unsigned segmentsPerRank = (static_cast<unsigned int>(rank) < unmatchedRows) ? (quotient+1) : quotient; 

    // Compute the offset in terms of rows from 0
    int row_offset = 0;
    for(int r = 0; r < rank; ++r)
        row_offset += (r < unmatchedRows) ? (quotient+1) : quotient;

    // Define effective number of rows (points) as (number of segments +1) +2 ghost rows
    unsigned local_nx = (segmentsPerRank + 1) + 2;
    
    // Define effective number of columns (points) as ny+1 --> same for every ran but called local for uniformity
    unsigned local_ny = ny + 1;

    //______________________________________________________________________________________

    // --- Initialize local grid data ---
    Eigen::ArrayXXd local_f   {local_nx-2, local_ny};
    Eigen::ArrayXXd local_u_ex{local_nx-2, local_ny}; // <-- va ricostruita alla fine per lo studio di convergenza

    // --- Fill grid data ---
    
    // Update all the rows of local forcing term and local exact solution
    #pragma omp parallel for collapse(2) schedule(static)
    for(unsigned i = 0; i < local_nx-2; ++i){
        for(unsigned j = 0; j < local_ny; ++j){
            // Update directly for rank 0
            if(rank==0){
                local_f   (i, j) = force    (bl[0] + i*h, bl[1] + j*h);
                local_u_ex(i, j) = exact_sol(bl[0] + i*h, bl[1] + j*h);
            // Consider the offsed in the grid along x axis for other ranks
            }else{
                local_f   (i, j) = force    (bl[0] + (row_offset-1 + i)*h, bl[1] + j*h);
                local_u_ex(i, j) = exact_sol(bl[0] + (row_offset-1 + i)*h, bl[1] + j*h);
            }
        }
    }

    // --- Initialize the local final solution matrix with ghost rows ---
    Eigen::ArrayXXd local_U0{Eigen::ArrayXXd::Zero(local_nx, local_ny)};
    Eigen::ArrayXXd local_U {Eigen::ArrayXXd::Zero(local_nx, local_ny)};

    // --- Apply boundary conditions to our candidate solution ---

    // Rank 0 : apply bottom boundary condition
    if(rank == 0)
        local_U0.row(1) = local_u_ex.row(0);
    

    // Rank size-1 : apply top boundary condition
    if(rank == size-1)
        local_U0.row(local_nx-2) = local_u_ex.row(local_nx-3);
    
    
    // Every rank : apply lateral condition    
    for(unsigned i = 1; i < local_nx-1; ++i){
        local_U0(i,0) = local_u_ex(i-1,0);
        local_U0(i,local_ny-1) = local_u_ex(i-1,local_ny-1);
    }

    if(rank==0){
        print_var("localU0", local_U0);
        print_var("localUex", local_u_ex);
    }

    //______________________________________________________________________________________

    // --- Initialize the solver ---
    Operator::Laplacian laplacian( 
        /*local_f    = */ local_f,
        /*local_u_ex = */ local_u_ex,
        /*h     = */ h,
        /*rank  = */ rank,
        /*size  = */ size,
        /*nx    = */ local_nx,
        /*nx    = */ local_ny
    );

    // --- Solve ---
    int global_converged = 0;
    int local_converged  = 0;
    unsigned k;

    for(k = 0; k < maxIt; ++k){
        // Reset local convergence
        local_converged = 0; 

        // --- Send and receive necessary data of the common rows ---

        // Exchange with upper rank --> send my top row, in the upper ghost row
        if(rank < size-1)
            MPI_Sendrecv(
                local_U0.row(local_nx-2).data(), local_ny, MPI_DOUBLE, rank+1, 0,
                local_U0.row(local_nx-1).data(), local_ny, MPI_DOUBLE, rank+1, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

        // Exchange with lower rank --> send my bottom row, in the lower ghost row
        if(rank > 0)
            MPI_Sendrecv(
                local_U0.row(1).data(), local_ny, MPI_DOUBLE, rank-1, 1,
                local_U0.row(0).data(), local_ny, MPI_DOUBLE, rank-1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

        //  --- Apply the stencil to the local grid ---
        laplacian(local_U0, local_U);

        // --- Check global convergence ---

        // Retreive the internal part for which I want to test convergence
        Eigen::ArrayXXd diff = local_U.block(1, 0, local_nx-2, local_ny) - local_U0.block(1, 0, local_nx-2, local_ny);

        // Check local convergence (su cosa va testata? tol comune o va divisa?)
        if(std::sqrt(h*diff.square().sum()) < tol)
            local_converged = 1;
        
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
        print_var("k all'uscita ", k);

        if(!global_converged){
            std::cout << "Jacobi iterations did not converge !" << std::endl;
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
        for(unsigned i = 0; i < size; ++i) {
            int rows_i = (i < unmatchedRows) ? (quotient+1) : quotient;
            recvcounts[i] = (rows_i + 1) * (ny+1);
            offsets[i] = sum;
            sum += recvcounts[i];
        }
    }    

    // Gatherv just the internal part of U
    MPI_Gatherv(local_U.data() + local_ny, (local_nx-2)*local_ny, MPI_DOUBLE, global_U.data(), 
                recvcounts.data(), offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //______________________________________________________________________________________    

    // --- Print the results ---

    if(rank==0){
        // Display the solution 
        print_var("Global U", global_U);

        // Print number of steps in which convergence is reached 
        print_var("Convergence reached in n steps", k);
    }
        
    // Close MPI environment
    MPI_Finalize();

    return 0;
}



