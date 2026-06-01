#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include <Eigen/Dense>
#include <omp.h>
#include <iostream>

namespace Operator{

/// @brief Functor for applying stencil with dirichlet-type BCs
class Laplacian{

    private:
        const Eigen::VectorXd bcTop;
        const Eigen::VectorXd bcBottom;
        const Eigen::VectorXd bcLeft;
        const Eigen::VectorXd bcRight;
        const Eigen::ArrayXXd f;
        const Eigen::ArrayXXd u_ex;
        const double h2;
        const unsigned nx;
        const unsigned ny;
        const int rank;
        const int size; // the same for all the instances
        unsigned i_start;
        unsigned i_end;
    
    public:
        /// @brief Constructor
        /// @param f_    matrix Eigen::ArrayXd for forcing term BCs
        /// @param u_ex_    matrix Eigen::ArrayXd for exact solution to compute boundary conditions
        /// @param h_    scalar double for the (uniform) mesh size
        /// @param rank_ scalar rank to differentiate Jacobi's updates
        /// @param size_ scalar size to differentiate Jacobi's updates
        /// @param nx_   scalar local nx --> number of rows of destination matrix
        /// @param ny_   scalar local ny --> number of columns of destination matrix

        Laplacian(
            const Eigen::ArrayXXd& f_,
            const Eigen::ArrayXXd& u_ex_,
            const double h_,
            const int rank_,
            const int size_,
            const unsigned nx_,
            const unsigned ny_
        ) : 
            f{f_},
            u_ex{u_ex_},
            h2{h_*h_},
            nx{nx_},
            ny{ny_},
            rank{rank_},
            size{size_}
            { 
                // Rank 0 : do not update first row
                i_start = (rank==0) ? 2 : 1;

                // Rank size-1 : do not update last row
                i_end = (rank==size-1) ? nx-3 : nx-2;
            }

        void operator()(const Eigen::ArrayXXd& src, Eigen::ArrayXXd& dest){
            
            // Parallel for to apply the stencil implementation
            #pragma omp parallel for collapse(2) schedule(static)
            for(unsigned i = i_start; i <= i_end; ++i)
                for(unsigned j = 1; j < ny-1; ++j)
                    dest(i,j) = 0.25*(src(i-1,j) + src(i+1,j) + src(i,j-1) + src(i,j+1) + h2*f(i-1,j));

            // --- Apply boundary conditions ---
           
            // Every rank : apply boundary condition on x (y fixed)
            for(unsigned i = 1; i < nx-1; ++i){
                dest(i,0)    = u_ex(i-1,0);
                dest(i,ny-1) = u_ex(i-1,ny-1);
            }

            // Rank 0 : apply boundary condition on y (x=0)
            if(rank == 0)
                for(unsigned j = 0; j < ny; ++j)
                     dest(1,j) = u_ex(0,j);           
            
            // Rank size-1 : apply boundary condition on y (x fixed)
            if(rank == size-1)
                for(unsigned j = 0; j < ny; ++j)
                    dest(nx-2, j) = u_ex((nx-2)-1, j);
            
        }
};

}

#endif