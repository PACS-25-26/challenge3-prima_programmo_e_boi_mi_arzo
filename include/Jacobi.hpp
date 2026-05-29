#pragma once

#include <Eigen/Dense>

namespace Solvers{

/// @brief Template class for a jacobi solver
/// @tparam Op    template argument for a general operator
/// @tparam Src   old solution(Eigen type for the moment)
/// @tparam Dest  new solution(Eigen type for the moment)
template <typename Op, typename Src, typename Dest>
class Jacobi{
    private:
        unsigned maxIt;
        double tol;
        double h;

    public:
        Jacobi(const unsigned maxIt_, const double tol_, const double h_) :
            maxIt{maxIt_},
            tol{tol},
            h{h_}
            {}

        void solve(const Op& op, Src& src, Dest& dest){

            // Convergence flag 
            bool converged = false;

            // Parallel loop 
            #pragma parallel for
            for(unsigned k = 0; k < maxIt; ++k){
                op(src, dest);

                if(std::sqrt(h*(dest - src).square().sum()) < tol){
                    converged = true;
                    break;
                }

                src = dest;
            }
        }

        // Setters
        void set_maxIt(const unsigned maxIt_) { maxIt = maxIt_; }
        void set_tol(const double tol_) { tol = tol_; }
        void set_h(const double h_) { h = h_; }
};

}