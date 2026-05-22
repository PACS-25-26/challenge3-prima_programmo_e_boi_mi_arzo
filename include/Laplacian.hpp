#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include <Eigen/Dense>
#include <omp.h>

namespace Operator{

/// @brief  Template class to apply the laplacian stencil with ceneterd finite differences of order 2
/// @tparam Nx Number of subinterval in direction x
/// @tparam Ny Number of subinterval in direction y
template <int Nx, int Ny>
class Laplacian{

    private:
        const Eigen::Array<double, Nx+1>       bcTop;
        const Eigen::Array<double, Nx+1>       bcBottom;
        const Eigen::Array<double, Ny+1>       bcLeft;
        const Eigen::Array<double, Ny+1>       bcRight;
        const Eigen::Array<double, Nx+1, Ny+1> f;
        const double h2;
    
    public:
        Laplacian(
            const Eigen::Array<double, Nx+1>& bcT,
            const Eigen::Array<double, Nx+1>& bcB,
            const Eigen::Array<double, Ny+1>& bcL,
            const Eigen::Array<double, Ny+1>& bcR,
            const Eigen::Array<double, Nx+1, Ny+1>& f_,
            const double h_
        ) : bcTop(bcT),
            bcBottom(bcB),
            bcLeft(bcL),
            bcRight(bcR),
            h2(h_*h_)
            {}

        void operator()(const Eigen::Array<double, Nx+1, Ny+1>& src, Eigen::Array<double, Nx+1, Ny+1>& dest){

            // Apply the stencil
            dest.template block<Nx-1, Ny-1>(1, 1) = 
                (0.25/h2)*(src.template block<Nx-1, Ny-1>(0, 1) // top
                + src.template block <Nx-1, Ny-1>(1, 0)        // left
                + src.template block <Nx-1, Ny-1>(1, 2)        // right
                + src.template block <Nx-1, Ny-1>(2, 1)        // bottom
                + f.template block   <Nx-1, Ny-1>(1,1));       // forcing term


            // Enforce boundary condition(Dirichlet type)
            dest.row(0)             = bcBottom;
            dest.row(dest.rows()-1) = bcTop;
            dest.col(0)             = bcLeft;
            dest.col(dest.cols()-1) = bcRight;

            // // Alternatively a parallel implementation
            // #pragma parallel for
            // for(Eigen::Index i = 1; i <= Nx-1; ++i)
            //     for(Eigen::Index j = 1; j <= Ny-1; ++j)
            //         dest(i,j) = (0.25/(h*h))*(src(i-1,j) + src(i+1,j) + src(i,j-1) + src(i,j+1) + f(i,j));
        }
};

}

#endif