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
    
    public:
        /// @brief Constructor
        /// @param bcT row Eigen::ArrayXd for top BCs
        /// @param bcB row Eigen::ArrayXd for bottom BCs
        /// @param bcL column Eigen::ArrayXd for left BCs
        /// @param bcR column Eigen::ArrayXd for right BCs
        /// @param f_  matrix Eigen::ArrayXd for forcing term BCs
        /// @param h_  scalar double for the (uniform) mesh size
        Laplacian(
            const Eigen::RowVectorXd& bcT,
            const Eigen::RowVectorXd& bcB,
            const Eigen::VectorXd&    bcL,
            const Eigen::VectorXd&    bcR,
            const Eigen::ArrayXXd&    f_,
            const double h_
        ) : bcTop{bcT},
            bcBottom{bcB},
            bcLeft{bcL},
            bcRight{bcR},
            f{f_},
            h2{h_*h_},
            nx{bcB.cols()},
            ny{bcL.rows()}
            {}

        void operator()(const Eigen::ArrayXXd& src, Eigen::ArrayXXd& dest){

            // Parallel for to apply the stencil implementation
            #pragma omp parallel for collapse(2) schedule(static)
            for(int i = 1; i <= nx-1; ++i)
                for(int j = 1; j <= ny-1; ++j)
                    dest(i,j) = 0.25*(src(i-1,j) + src(i+1,j) + src(i,j-1) + src(i,j+1) + h2*f(i,j));

            std::cout << "éééééééééééé" << std::endl;


            // Enforce boundary condition(Dirichlet type)
            dest.row(0)             = bcBottom;
            dest.row(dest.rows()-1) = bcTop;
            dest.col(0)             = bcLeft;
            dest.col(dest.cols()-1) = bcRight;

            std::cout << "éééééééééééé" << std::endl;
         
        }
};

}

#endif