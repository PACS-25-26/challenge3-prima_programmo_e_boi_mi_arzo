
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <numbers>
#include <omp.h>
#include <mpi.h>
#include <chrono>

#include "include/Parameters.hpp"
#include "include/Laplacian.hpp"
#include "include/PrintVar.hpp"
#include "include/Data.hpp"

int main(int argc, char** argv){
    
    // Alias __________________________________________________________________________________
    using ArrayXXd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic/*, Eigen::RowMajor*/>;

    // MPI setup ______________________________________________________________________________
    int provided;
    // Request FUNNELED mode
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Error: MPI library does not provide enough thread support!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read thread count from arguments
    int num_threads = 1;
    if (argc >= 2) {
        num_threads = std::atoi(argv[1]);
    }
    omp_set_num_threads(num_threads);

    // Get rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    // Check the number of processors and threads
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        #pragma omp critical
        {
            std::cout << "[MPI Rank " << rank << "/" << size 
                      << "] Thread " << thread_id << "/" << num_threads 
                      << " is running." << std::endl;
        }
    }

    // Data ___________________________________________________________________________________
    // Domain parameters
    int nx, ny;
    double h;
    constexpr Test::Point bl = Test::bottom_left;
    constexpr Test::Point tr = Test::top_right;
    
    // Solver parameters
    int maxIt;
    double tol;

    // Physical data
    Test::ForcingTerm   force;
    Test::ExactSolution exact_sol;

    // --- Parameters reading made by rank 0 only ---
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
    MPI_Bcast(&nx,    1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&ny,    1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&maxIt, 1, MPI_INT,    0, MPI_COMM_WORLD);
    MPI_Bcast(&h,     1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tol,   1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Data splitting parameters ______________________________________________________________
   
    // --- Compute number of cols assigned to each rank balancing eccess ---
    MPI_Barrier(MPI_COMM_WORLD);
    int evenCols        = (ny+1) / size;
    int unmatchedCols   = (ny+1) % size - 1;
    int baseColsPerRank = (static_cast<int>(rank) < unmatchedCols) ? (evenCols+1) : evenCols; 
    // std::cout << "rank = "<< rank << ", evenCols = " << evenCols << std::endl;
    // std::cout << "rank = "<< rank << ", unmatchedCols = " << unmatchedCols << std::endl; 
    // std::cout << "rank = "<< rank << ", baseColsPerRank = " << baseColsPerRank << std::endl; 

    // --- Compute the offset in terms of cols from 0 ---
    int col_offset = 0;
    for(int r = 0; r < rank; ++r)
        col_offset += (r < unmatchedCols) ? (evenCols+1) : evenCols;
    if(rank != 0)
        col_offset -= 1;
    // std::cout << "rank = "<< rank << ", col_offset = " << col_offset << std::endl; 

    // --- Define effective number of cols (points) ---
    int local_cols = (rank != 0) ? (baseColsPerRank + 2) : (baseColsPerRank + 1);
    // std::cout << "rank = "<< rank << ", local_cols = " << local_cols << std::endl; 

    // --- Define the number of columns to update ---
    int local_cols_to_update = local_cols - 2;
    // std::cout << "rank = "<< rank << ", local_cols_to_update = " << local_cols_to_update << std::endl; 
    
    // --- Define effective number of rows (points) as nx+1(same for every rank but called local for uniformity) ---
    int local_rows = nx + 1;
    // std::cout << "rank = "<< rank << ", local_rows = " << local_rows << std::endl; 

    // --- Define the number of rows to update ---
    int local_rows_to_update = local_rows - 2;
    std::cout << "rank = "<< rank << ", local_rows_to_update = " << local_rows_to_update << std::endl; 

    // Initialization of physical data ____________________________________________________________

    // --- Initialize local grid data ---
    ArrayXXd local_f   {local_rows_to_update, local_cols_to_update};
    ArrayXXd local_u_ex{local_rows_to_update, local_cols_to_update}; // <-- va ricostruita alla fine per lo studio di convergenza

    // --- Fill local forcing term and local exact solution ---
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = 0; i < local_f.rows(); ++i){
        for(int j = 0; j < local_f.cols(); ++j){
            local_f   (i, j) = force    (bl[0] + (i+1)*h, bl[1] + (col_offset+j+1)*h);
            local_u_ex(i, j) = exact_sol(bl[0] + (i+1)*h, bl[1] + (col_offset+j+1)*h);
        }
    }

    // --- Check correctness ---
    print_line();
    for(int r = 0; r < size; ++r){
        if(rank == r){
            std::cout << "rank = "<< rank << ", local_f = " << local_f << std::endl;
            std::cout << "rank = "<< rank << ", local_uex = " << local_u_ex << std::endl; 
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank == 0){
        print_line();
        ArrayXXd uex_test{nx+1, ny+1};
        ArrayXXd f_test{nx+1,ny+1};

        #pragma omp parallel for collapse(2) schedule(static)
        for(int i = 0; i < f_test.rows(); ++i){
            for(int j = 0; j < f_test.cols(); ++j){
                f_test  (i, j) = force    (bl[0] + i*h, bl[1] + j*h);
                uex_test(i, j) = exact_sol(bl[0] + i*h, bl[1] + j*h);
            }
        }

        print_var("uex", uex_test);
        print_var("f", f_test);

        print_line();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Initialize the local final solution matrix with ghost cols ---
    ArrayXXd local_U0{ArrayXXd::Zero(local_rows, local_cols)};
    ArrayXXd local_U {ArrayXXd::Zero(local_rows, local_cols)};

    // --- Apply boundary conditions to our candidate solution ---
    // Rank 0 : apply bottom boundary condition
    if(rank == 0){
        #pragma omp parallel for
        for(int i = 0; i < local_rows; ++i)
            local_U(i, 0) = exact_sol(bl[0]+i*h, bl[1]);
    }

    // Rank size-1 : apply top boundary condition
    if(rank == size-1){
        #pragma omp parallel for
        for(int i = 0; i < local_rows; ++i)
            local_U(i, local_cols-1) = exact_sol(bl[0]+i*h, tr[1]);
    }
    
    // Every rank : apply lateral condition
    #pragma omp parallel for 
    for(int j = 0; j < local_cols; ++j){
        // Left
        local_U(0,j) = exact_sol(bl[0], bl[1] + (col_offset+j)*h);

        // Right
        local_U(local_rows-1,j) = exact_sol(tr[0], bl[1] + (col_offset+j)*h);
    }

    // --- Set the initial guess ---
    local_U0 = local_U;

    // Solve ______________________________________________________________________________________

    // --- Initialize the solver ---
    Operator::Laplacian laplacian( 
        /*local_f = */ local_f,
        /*h       = */ h
    );

    // --- Solve ---
    // Diagnostic variables
    int global_converged = 0;
    int local_converged  = 0;
    int k;

    // Actual loop(this can be optimized moving the logic outside ---> more verbosity)
    for(k = 0; k < maxIt; ++k){
        // --- Reset local convergence ---
        local_converged = 0; 

        // --- Send and receive necessary data of the common rows ---
        // Exchange with upper rank --> send my last updated column, recieve the first updated column by the other rank
        if(rank < size-1)
            MPI_Sendrecv(
                local_U0.col(local_cols-2).data(), local_rows, MPI_DOUBLE, rank+1, 0,
                local_U0.col(local_cols-1).data(), local_rows, MPI_DOUBLE, rank+1, 1,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );
        // Exchange with lower rank --> send my first updated column, recieve the last updated column by the other rank
        if(rank > 0)
            MPI_Sendrecv(
                local_U0.col(1).data(), local_rows, MPI_DOUBLE, rank-1, 1,
                local_U0.col(0).data(), local_rows, MPI_DOUBLE, rank-1, 0,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE
            );

        // --- Apply the stencil to the local grid ---
        laplacian(local_U0, local_U);

        // --- Check global convergence ---
        // Retreive the internal part for which I want to test convergence
        const ArrayXXd diff = local_U.block(1, 1, local_rows_to_update, local_cols_to_update) - local_U0.block(1, 1, local_rows_to_update, local_cols_to_update);

        // Check local convergence
        if(std::sqrt(h*diff.square().sum()) < tol)
            local_converged = 1;

        // Check global convergence
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(&local_converged, &global_converged, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);

        // Checks convergence
        if(global_converged)
            break;

        // Update step
        local_U0.swap(local_U);
    }

    // If convergence was not reached within maxIt print a message
    if(rank == 0){
        print_var("Total number of iterations", k);

        if(!global_converged){
            std::cout << "Jacobi iterations did not converge !" << std::endl;
            return 1;
        }
        std::cout << "rank = "<< rank << ", local_U = " << local_U << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 1){
        std::cout << "rank = "<< rank << ", local_U = " << local_U << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 2){
        std::cout << "rank = "<< rank << ", local_U = " << local_U << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 3){
        std::cout << "rank = "<< rank << ", local_U = " << local_U << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Compute error in L2 norm _______________________________________________________________
    const ArrayXXd residual = local_U.block(1, 1, local_rows_to_update, local_cols_to_update) - local_u_ex;
    double local_squared_residual_sum = residual.square().sum();

    double global_squared_residual_sum;
    MPI_Reduce(
        &local_squared_residual_sum,
        &global_squared_residual_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        const double global_L2 = std::sqrt(h*global_squared_residual_sum);
        print_var("||u_ex - u_h||_L2", global_L2);
    }
        
    // Close MPI environment
    MPI_Finalize();

    return 0;
}



