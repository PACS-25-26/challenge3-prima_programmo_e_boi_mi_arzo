#pragma once

#include <Eigen/Dense>

/*
CONCEPTS ?
*/

// --- Namespace just to avoid conflicts in case we want to do an other one ---
namespace Test{

// Boundary condition object ______________________________________________________________________________________________________________
template <typename T>
struct BoundaryCondition{
    T operator()(const T& input){
        return Eigen::Zero(T.rows(), T.cols());
    }
};


// Forcing term object _____________________________________________________________________________________________________________________
// f(x,y) = // 8π2 sin(2πx) sin(2πy)
struct ForcingTerm{
    // --- Call operator using two linear algebra - like vectors ---
    Eigen::ArrayXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y){
        // Build the grid
        const Eigen::ArrayXd XY = (x * y.transpose()).array();

        // Return the value
        return (8.0*EIGEN_PI)*(2.0*EIGEN_PI*XY).sin()*(2.0*EIGEN_PI*XY).sin();
    }

    // --- Call operator passing the grid ---
    Eigen::ArrayXd operator()(const Eigen::ArrayXd& grid){
        return (8.0*EIGEN_PI)*(2.0*EIGEN_PI*grid).sin()*(2.0*EIGEN_PI*grid).sin();
    }
};


// Exact solution ___________________________________________________________________________________________________________________________
// u_ex(x,y) = sin(2πx) sin(2πy)
struct ExactSolution{
    // --- Call operator using two linear algebra - like vectors ---
    Eigen::ArrayXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y){
        const Eigen::ArrayXd XY = (x * y.transpose()).array();
        return (2.0*EIGEN_PI*XY).sin()*(2.0*EIGEN_PI*XY).sin();
    }

    // --- Call operator passing the grid ---
    Eigen::ArrayXd operator()(const Eigen::ArrayXd& grid){
        return (2.0*EIGEN_PI*grid).sin()*(2.0*EIGEN_PI*grid).sin();
    }
};


}