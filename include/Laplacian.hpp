#ifndef LAPLACIAN_HPP
#define LAPLACIAN_HPP

#include <Eigen/Dense>
#include <omp.h>
#include <iostream>

namespace Operator{

class Laplacian{

    using ArrayXXd = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    private:
        const Eigen::ArrayXXd f;
        const double h2;
    
    public:

        Laplacian(
            const ArrayXXd& f_,
            const double h_
        ) : 
            f{f_},
            h2{h_*h_}
            {}

        void operator()(const ArrayXXd& src, ArrayXXd& dest){
            
            // Parallel for to apply the stencil implementation
            #pragma omp parallel for collapse(2) schedule(static)
            for(unsigned i = 1; i < src.rows()-1; ++i)
                for(unsigned j = 1; j < src.cols()-1; ++j)
                    dest(i,j) = 0.25*(src(i-1,j) + src(i+1,j) + src(i,j-1) + src(i,j+1) + h2*f(i-1,j-1));
        }
};

}

#endif