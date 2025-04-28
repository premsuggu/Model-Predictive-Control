#pragma once
#include <casadi/casadi.hpp>
#include <vector>
#include <string>

// RK4 Discretization (creates symbolic expression)
casadi::MX rk4(const casadi::Function& f, casadi::MX x, casadi::MX u, double h);

// Cost Function (creates symbolic expression)
casadi::MX create_cost_function(
    const std::vector<casadi::MX>& xs,
    const std::vector<casadi::MX>& us,
    const casadi::MX& x_ref_p,
    const casadi::MX& u_ref_p,
    const casadi::MX& Q,
    const casadi::MX& R,
    const casadi::MX& Qf
);

// Dynamics Constraints (creates symbolic expression)
casadi::MX create_dynamics_constraints(
    const std::vector<casadi::MX>& xs,
    const std::vector<casadi::MX>& us,
    const casadi::Function& f,
    double h
);

// Simulate dynamics using RK4 (numerical)
casadi::DM simulate_dynamics(const casadi::Function& f, const casadi::DM& x, const casadi::DM& u, double h);

// Save results to CSV
void save_to_csv(const std::string& filename, const std::vector<std::vector<double>>& data);

// NMPCProblem struct
struct NMPCProblem {
    casadi::Function solver;
    int N, nx, nu;

    NMPCProblem(int N_, int nx_, int nu_, double h,
                const casadi::Function& f,
                const casadi::DM& Q, const casadi::DM& R, const casadi::DM& Qf);

    casadi::DM solve(const casadi::DM& x_init,
                    const casadi::DM& x_ref,
                    const casadi::DM& u_ref,
                    const casadi::DM& u_min,
                    const casadi::DM& u_max,
                    const casadi::DM& x_min,
                    const casadi::DM& x_max,
                    const casadi::DM* warm_start_x0 = nullptr);
};
