#pragma once

#include <numbers>
#include <array>

/*
CONCEPTS ?
*/

// --- Namespace just to avoid conflicts in case we want to do an other one ---
namespace Test{

constexpr double pi = std::numbers::pi;

// Domain parameters ______________________________________________________________________________________________________________________
using Point = std::array<double, 2>;
constexpr Point bottom_left = {0.0, 0.0};
constexpr Point top_right   = {1.0, 1.0};
constexpr int Nx = 16;
constexpr int Ny = 16;
constexpr double hx = (top_right[0] - bottom_left[0]) / static_cast<double>(Nx);
constexpr double hy = (top_right[1] - bottom_left[1]) / static_cast<double>(Ny);

// Boundary condition object ______________________________________________________________________________________________________________
struct BoundaryCondition{
    double operator()(const double x, const double y) const{
        return 0.0;
    }
};


// Forcing term object _____________________________________________________________________________________________________________________
// f(x,y) = // 8π2 sin(2πx) sin(2πy)
struct ForcingTerm{
    double operator()(const double x, const double y) const{
        return 8 * (pi*pi) * std::sin(2*pi*x) * std::sin(2*pi*y);
    }
};


// Exact solution ___________________________________________________________________________________________________________________________
// u_ex(x,y) = sin(2πx) sin(2πy)
struct ExactSolution{
    double operator()(const double x, const double y) const{
        return std::sin(2*pi*x)*std::sin(2*pi*y);
    }
};


}