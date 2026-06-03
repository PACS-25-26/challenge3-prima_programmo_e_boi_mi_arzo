
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

    using ArrayXXd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // Get rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Declare common variables
    int nx, ny, maxIt;
    double h, tol;

    // Declare boundary points and common functions
    constexpr Test::Point bl = Test::bottom_left;
    constexpr Test::Point tr = Test::top_right;
    Test::ForcingTerm       force;
    Test::ExactSolution     exact_sol;

  
    //______________________________________________________________________________________

    // --- Parameters reading made by rank 0---
    if (rank == 0){
        // Read the parameters from json
        const std::string FileName = "parameters.json";
        Parameters p(FileName);

        // Compute h (same for x and y as specified in the text)
        h = 1 / static_cast<double>(p.N);

        // Compute the number of subintervals in the two directions accordingly to h
        nx = static_cast<int>((tr[0] - bl[0])/h);
        ny = static_cast<int>((tr[1] - bl[1])/h);

        // Assign max number of iterations for the Broadcast
        maxIt = p.maxIt;

        // Assign tollerance for the Broadcast
        tol = p.tol;
    }

    // --- Rank 0 Broadcasts parameters to all the ranks ---
    MPI_Bcast(&nx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxIt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tol, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //______________________________________________________________________________________
   
    // Compute number of rows assigned to each rank balancing eccess
    int evenRows        = (nx+1) / size;
    int unmatchedRows   = (nx+1) % size;
    int baseRowsPerRank = (static_cast<int>(rank) < unmatchedRows) ? (evenRows+1) : evenRows; 

    // Compute the offset in terms of rows from 0
    int row_offset = 0;
    for(int r = 0; r < rank; ++r)
        row_offset += (r < unmatchedRows) ? (evenRows+1) : evenRows;

    // Define effective number of rows (points) as (number of segments +1) +2 ghost rows
    int local_rows = (rank == 0 || rank == size - 1) ? (baseRowsPerRank + 1) : (baseRowsPerRank + 2);
    
    // Define effective number of columns (points) as ny+1 --> same for every ran but called local for uniformity
    int local_cols = ny + 1;

    //______________________________________________________________________________________
    print_var("rank", rank);
    print_var("local_rows", local_rows);
    print_var("local_cols", local_cols);

    // --- Initialize local grid data ---
    ArrayXXd local_f   {local_rows-2, local_cols-2};
    ArrayXXd local_u_ex{local_rows-2, local_cols-2}; // <-- va ricostruita alla fine per lo studio di convergenza

    // --- Fill grid data ---
    
    // Update all the rows of local forcing term and local exact solution
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = 0; i < local_rows-2; ++i){
        for(int j = 0; j < local_cols-2; ++j){
            // Update directly for rank 0
            if(rank==0){
                local_f   (i, j) = force    (bl[0] + (i+1)*h, bl[1] + (j+1)*h);
                local_u_ex(i, j) = exact_sol(bl[0] + (i+1)*h, bl[1] + (j+1)*h);
            // Consider the offset in the grid along x axis for other ranks
            }else{
                local_f   (i, j) = force    (bl[0] + (row_offset + i)*h, bl[1] + (j+1)*h);
                local_u_ex(i, j) = exact_sol(bl[0] + (row_offset + i)*h, bl[1] + (j+1)*h);
            }
        }
    }

    // --- Initialize the local final solution matrix with ghost rows ---
    ArrayXXd local_U0{ArrayXXd::Zero(local_rows, local_cols)};
    ArrayXXd local_U {ArrayXXd::Zero(local_rows, local_cols)};

    // --- Apply boundary conditions to our candidate solution ---

    // Rank 0 : apply bottom boundary condition
    if(rank == 0)
        for(int j = 0; j < local_cols; ++j)
            local_U(0, j) = exact_sol(bl[0], bl[1] + j*h);

    

    // Rank size-1 : apply top boundary condition
    if(rank == size-1)
        for(int j = 0; j < local_cols; ++j)
            local_U(local_rows-1, j) = exact_sol(tr[0], bl[1] + j*h);
    
    
    // Every rank : apply lateral condition    
    for(int i = 1; i < local_rows-1; ++i){
        local_U(i,0)            = exact_sol(bl[0] + (row_offset + i)*h, bl[1]);
        local_U(i,local_cols-1) = exact_sol(bl[0] + (row_offset + i)*h, tr[1]);
    }

    local_U0 = local_U;

    // if(rank==0){
    //     print_var("localU0", local_U0);
    //     print_var("localUex", local_u_ex);
    // }

    print_var("rank----------------------------", rank);
    print_var("localU0", local_U0);
    print_var("localUex", local_u_ex);

    //______________________________________________________________________________________

    // --- Initialize the solver ---
    Operator::Laplacian laplacian( 
        /*local_f    = */ local_f,
        /*h     = */ h
    );

    // --- Solve ---
    int global_converged = 0;
    int local_converged  = 0;
    int k;

    for(k = 0; k < maxIt; ++k){
        // Reset local convergence
        local_converged = 0; 

        // --- Send and receive necessary data of the common rows ---

        // Exchange with upper rank --> send my top row, in the upper ghost row
        if(rank < size-1)
            MPI_Sendrecv(
                local_U0.row(local_rows-2).data(), local_cols, MPI_DOUBLE, rank+1, 0,
                local_U0.row(local_rows-1).data(), local_cols, MPI_DOUBLE, rank+1, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        // Exchange with lower rank --> send my bottom row, in the lower ghost row
        if(rank > 0)
            MPI_Sendrecv(
                local_U0.row(1).data(), local_cols, MPI_DOUBLE, rank-1, 1,
                local_U0.row(0).data(), local_cols, MPI_DOUBLE, rank-1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        //  --- Apply the stencil to the local grid ---
        laplacian(local_U0, local_U);

        // --- Check global convergence ---
        // Retreive the internal part for which I want to test convergence
        ArrayXXd diff = local_U - local_U0;

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
    ArrayXXd global_U;

    // Prepare to receive global U
    std::vector<int> recvcounts(size, 0);
    std::vector<int> offsets(size, 0);

    if(rank == 0){
        // Resize it only for rank 0
        global_U.resize(nx+1, ny+1);

        // int sum = 0;

        // // Keep trace of how many elements each process has
        // for(int i = 0; i < size; ++i) {
        //     // int rows_i    = (i < unmatchedRows) ? (evenRows+1) : evenRows;
        //     recvcounts[i] = (rows_i + 1) * (ny+1);
        //     offsets[i]    = sum;
        //     sum          += recvcounts[i];
        // }
    }
    // Gather for recvcounts
    int local_elements_to_send = (local_rows-2)*local_cols;
    std::cout << "rank = " << rank << ", local_el = " << local_elements_to_send << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(
        &local_elements_to_send, 1, MPI_INT,
        recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD
    );

    int offset_to_send = 0; 
    for(int i = 0; i < rank; ++i)
        offset_to_send += recvcounts[i];

    std::cout << "rank = " << rank << ", offset finale = " << offset_to_send << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allgather(
        &offset_to_send, 1, MPI_INT,
        offsets.data(),  1, MPI_INT, MPI_COMM_WORLD
    );

    // Gatherv just the internal part of U
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(
        local_U.data() + local_cols, local_elements_to_send, MPI_DOUBLE,
        global_U.data() + local_cols, recvcounts.data(), offsets.data(), MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    //______________________________________________________________________________________    

    // --- Print the results ---

    if(rank==0){
        // Display the solution 
        print_var("Global U", global_U);

        // Print number of steps in which convergence is reached 
        print_var("Convergence reached in n steps", k);

        ArrayXXd Uex{nx+1, ny+1};

        #pragma omp parallel for collapse(2) schedule(static)
        for(int i = 0; i < nx+1; ++i)
            for(int j = 0; j < ny+1; ++j)
                Uex(i,j) = exact_sol(bl[0]+i*h, bl[1]+j*h);

        print_var("Uex", Uex);

    }
        
    // Close MPI environment
    MPI_Finalize();

    return 0;
}



