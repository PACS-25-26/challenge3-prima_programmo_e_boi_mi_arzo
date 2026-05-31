#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include <Eigen/Dense>
#include <omp.h>
#include <iostream>

namespace Operator{

/// @brief Functor for applying stencil with dirichlet-type BCs
class Laplacian{

    private:
        const Eigen::RowVectorXd bcTop;
        const Eigen::RowVectorXd bcBottom;
        const Eigen::VectorXd    bcLeft;
        const Eigen::VectorXd    bcRight;
        const Eigen::ArrayXXd    f;
        const double h2;
        const unsigned nx;
        const unsigned ny;
        const int rank;
        const int size; // the same for all the instances
        unsigned i_start;
        unsigned i_end;
    
    public:
        /// @brief Constructor
        /// @param bcT   row Eigen::ArrayXd for top BCs
        /// @param bcB   row Eigen::ArrayXd for bottom BCs
        /// @param bcL   column Eigen::ArrayXd for left BCs
        /// @param bcR   column Eigen::ArrayXd for right BCs
        /// @param f_    matrix Eigen::ArrayXd for forcing term BCs
        /// @param h_    scalar double for the (uniform) mesh size
        /// @param rank_ scalar rank to differentiate Jacobi's updates
        /// @param size_ scalar size to differentiate Jacobi's updates
        /// @param nx_   scalar local nx
        Laplacian(
            const Eigen::RowVectorXd& bcT,
            const Eigen::RowVectorXd& bcB,
            const Eigen::VectorXd&    bcL,
            const Eigen::VectorXd&    bcR,
            const Eigen::ArrayXXd&    f_,
            const double h_,
            const int rank_,
            const int size_,
            const unsigned nx_
        ) : bcTop{bcT},
            bcBottom{bcB},
            bcLeft{bcL},
            bcRight{bcR},
            f{f_},
            h2{h_*h_},
            nx{nx_},
            ny{bcL.rows()-1},
            rank{rank_},
            size{size_}
            { 
                // Rank 0 : do not update first row
                i_start = (rank==0) ? 2 : 1;

                // Rank size-1 : do not update last row
                i_end = (rank==size-1) ? nx-1 : nx;
            }

        void operator()(const Eigen::ArrayXXd& src, Eigen::ArrayXXd& dest){
            
            // Parallel for to apply the stencil implementation
            #pragma omp parallel for collapse(2) schedule(static)
            for(unsigned i = i_start; i <= i_end; ++i)
                for(unsigned j = 1; j <= ny; ++j)
                    dest(i,j) = 0.25*(src(i-1,j) + src(i+1,j) + src(i,j-1) + src(i,j+1) + h2*f(i,j));

            // --- Apply boundary conditions ---

            // Every rank : apply lateral condition
            dest.col(0)             = bcLeft;
            dest.col(dest.cols()-1) = bcRight;

            // Rank 0 : apply bottom boundary condition
            if(rank == 0)
                dest.row(1) = bcBottom;

            // Rank size-1 : apply top boundary condition 
            if(rank == size-1)
                dest.row(nx) = bcTop;

        }
};

}

#endif