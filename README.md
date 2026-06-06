[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tKSbaXxd)

# 2D Poisson Equation Solver (Hybrid MPI + OpenMP)

This project implements a parallel solver for the 2D Poisson equation using the Finite Difference Method (FDM) and the Jacobi iterative solver. The parallelization strategy employs a hybrid approach, utilizing **MPI** for distributed memory (1D column-wise domain decomposition) and **OpenMP** for shared memory multithreading within each MPI rank.

## Mathematical Formulation
The solver approximates the solution to the problem:
- Domain: $\Omega = [0, 1] \times [0, 1]$
- Exact Solution: $u(x,y) = \sin(2\pi x) \sin(2\pi y)$
- Forcing Term: $f(x,y) = 8\pi^2 \sin(2\pi x) \sin(2\pi y)$
- Boundary Conditions: Dirichlet (extracted from the exact solution)

## Dependencies
To compile and run this project, the following libraries and tools are required:
* **C++ Compiler** with C++20 support (e.g., `g++` or `clang++`)
* **MPI** (e.g., OpenMPI or MPICH)
* **OpenMP**
* **Eigen3**: For matrix operations and grid representation.
* **nlohmann/json**: For reading configuration files.

## Project Structure
* `main.cpp`: Main entry point containing the MPI setup, domain decomposition, communication of ghost cells (`MPI_Sendrecv`), and the main Jacobi loop.
* `include/Parameters.hpp`: Handles parsing of the `parameters.json` configuration file.
* `include/Data.hpp`: Contains the physical domain definition, exact solution, and forcing term.
* `include/Laplacian.hpp`: Functor that applies the 5-point finite difference stencil.
* `include/PrintVar.hpp`: Utility functions for printing variables and matrices.

## Configuration
The solver reads its execution parameters from a `parameters.json` file located in the working directory. 

Example `parameters.json`:
```json
{
    "N_ref": 4,
    "tol": 1e-8,
    "maxIt": 1000000
}
```

* `N_ref`: Determines the grid resolution. The number of intervals $N$ is computed as $N = 2^{N\_ref}$.
* `tol`: The tolerance for the $L^2$ norm of the residual to determine global convergence.
* `maxIt`: The maximum number of iterations allowed for the Jacobi solver.

## Compilation
This project uses a `Makefile` to handle compilation and dependencies. Ensure that the required libraries (Eigen3, nlohmann/json, and OpenMPI) are installed on your system. 

To compile the project, simply run:
```bash
make
```

## Additional Make Targets
The Makefile also provides the following utility commands:
* make clean: Removes object files (.o) and the dependency tracking file.
* make distclean: Performs a deep clean, removing the executable, .csv output files, and generated documentation
* make doc: Automatically generates project documentation using Doxygen.

## Usage
Run the generated main executable using mpirun (or mpiexec). You can optionally pass the number of OpenMP threads as a command-line argument. If not provided, the solver defaults to 1 thread per MPI process.

```bash
Run with 4 MPI processes and 2 OpenMP threads per process
mpirun -np 4 ./main 2
```

## Output
The program will output:
1. Thread initialization diagnostics for each rank.
2. The total time elapsed for the computation on each rank (measured using `std::chrono`).
3. The total number of Jacobi iterations performed (printed by rank 0).
4. The global $L^2$ error norm $||u_{ex} - u_h||_{L^2}$ computed against the exact analytical solution.

**Example Output:**
```text
[MPI Rank 0/2] Thread 0/2 is running.
[MPI Rank 0/2] Thread 1/2 is running.
[MPI Rank 1/2] Thread 0/2 is running.
[MPI Rank 1/2] Thread 1/2 is running.
Correctly read the json file and set all the parameters
Total elapsed time on rank 1 = 6.24955
Total elapsed time on rank 0 = 6.2486
Total number of iterations = 39990

||u_ex - u_h||_L2 = 0.000354678
