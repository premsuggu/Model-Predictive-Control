#pragma once
#include <casadi/casadi.hpp>
#include <vector>
#include <string>

// ---------------- RK4 Discretization (symbolic expression) -------------------
casadi::MX rk4(const casadi::Function& f, casadi::MX& x, casadi::MX u, double h);

// ------------------------ NMPC Problem Definition --------------------------
casadi::MX create_dynamics_constraints(
    const std::vector<casadi::MX>& xs,
    const std::vector<casadi::MX>& us,
    const casadi::Function& f,
    double h
);

// ------------------------ Cost Function (symbolic expression) --------------------------
casadi::MX create_cost_function(
    const std::vector<casadi::MX>& xs,
    const std::vector<casadi::MX>& us,
    const casadi::MX& x_ref,
    const casadi::MX& u_ref,
    const casadi::MX& Q,
    const casadi::MX& R,
    const casadi::MX& Qf
);

//------------------------- Solve the NMPC Problem --------------------------
casadi::DM solve_nlp(
    int N,
    double h,
    int nx,
    int nu,
    const casadi::Function& f,
    const casadi::DM& x_init,
    const casadi::DM& x_ref,
    const casadi::DM& u_ref,
    const casadi::DM& Q,
    const casadi::DM& R,
    const casadi::DM& Qf,
    const casadi::DM& u_min,
    const casadi::DM& u_max,
    const casadi::DM& x_min,
    const casadi::DM& x_max,
    const casadi::DM* warm_start_x0
);

//-------------------Simulate dynamics using RK4-------------------------
casadi::DM simulate_dynamics(const casadi::Function& f, const casadi::DM& x, const casadi::DM& u, double h);

//---------------------------Save results to CSV-----------------------------
void save_to_csv(const std::string& filename, const std::vector<std::vector<double>>& data);
