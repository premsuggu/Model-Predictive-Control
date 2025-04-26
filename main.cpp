#include "nmpc_utils.hpp"
#include <casadi/casadi.hpp>
#include <iostream>

int main() {
    using casadi::sin;
    using casadi::cos;

    int N = 20;     //prediction horizon
    int nx = 4;
    int nu = 1;
    double h = 0.1;
    int sim_steps = 100;

    //Physical parameters for pendulum cart system 
    double M = 1.0;     // mass of cart
    double m = 0.5;     // mass of pendulum
    double l = 0.6;     // length to pendulum center of mass
    double g = 9.81;    // gravity


    // Dynamics (pendulum on cart)
    casadi::MX x = casadi::MX::sym("x", nx);    // state vector [x, x_dot, theta, theta_dot]
    casadi::MX u = casadi::MX::sym("u", nu);    // control input (force on cart)

    casadi::MX p = x(0);
    casadi::MX v = x(1);
    casadi::MX theta = x(2);
    casadi::MX omega = x(3);

    // Intermediate computations
    casadi::MX sin_theta = sin(theta);
    casadi::MX cos_theta = cos(theta);
    casadi::MX total_mass = M + m;

    casadi::MX temp = (u(0) + m * l * omega * omega * sin_theta) / total_mass;

    casadi::MX theta_dd_num = g * sin_theta - cos_theta * temp;
    casadi::MX theta_dd_den = l * (4.0/3.0 - m * cos_theta * cos_theta / total_mass);
    casadi::MX theta_dd = theta_dd_num / theta_dd_den;
    
    casadi::MX v_dd = temp - m * l * theta_dd * cos_theta / total_mass;
    
    casadi::MX x_dot = casadi::MX::vertcat({
        v,
        v_dd,
        omega,
        theta_dd
    });
    
    casadi::Function f("f", {x, u}, {x_dot});

    casadi::DM x_ref = casadi::DM::zeros(nx, N+1);          // Reference trajectory
    
    for (int k = 0; k <= N; ++k) {        
        x_ref(0, k) = 1.5; 
        x_ref(1, k) = 0.0;
        x_ref(2, k) = 0.0;
        x_ref(3, k) = 0.0;          
    }


    casadi::DM u_ref = casadi::DM::zeros(nu);               // desired control set to zero
    casadi::DM Q = casadi::DM::eye(nx);                     // state cost matrix
    Q(0,0) = 1; Q(1,1) = 1; Q(2,2) = 10; Q(3,3) = 1; 
    casadi::DM R = 0.5 * casadi::DM::eye(nu);               // control cost matrix
    casadi::DM Qf =  2*Q;                                   // terminal cost matrix

    casadi::DM x_curr = casadi::DM::zeros(nx);
    x_curr(0) = 0.0; x_curr(1) = 0.0; x_curr(2) = 0.0; x_curr(3) = 0.0;     // Initial state

    casadi::DM x_min = casadi::DM::zeros(nx);   
    casadi::DM x_max = casadi::DM::zeros(nx);
    
    x_min(0) = -5.0;   x_max(0) = 5.0;              // cart position limits (meters)
    x_min(1) = -3.0;   x_max(1) = 3.0;              // velocity
    x_min(2) = -0.5;   x_max(2) = 0.5;              // angle limits (radians)
    x_min(3) = -2.0;   x_max(3) = 2.0;              // angular velocity
    
    casadi::DM u_min = casadi::DM::zeros(nu);
    casadi::DM u_max = casadi::DM::zeros(nu);
    u_min(0) = -3.0;
    u_max(0) = 3.0;
        
    // state bounds

    std::vector<std::vector<double>> state_traj;
    std::vector<std::vector<double>> control_traj;

    casadi::DM prev_sol; 
    for (int t = 0; t < sim_steps; ++t) {

        casadi::DM* warm_start_ptr = nullptr;
        if (!prev_sol.is_empty()) {
            warm_start_ptr = &prev_sol;
        }
    
        casadi::DM sol = solve_nlp(
            N, h, nx, nu, f, x_curr,
            x_ref, u_ref, Q, R, Qf,
            u_min, u_max, x_min, x_max,
            warm_start_ptr
        );

        prev_sol = sol;

        casadi::DM u0 = sol(casadi::Slice((N+1)*nx, (N+1)*nx + nu));
        casadi::DM x_next = simulate_dynamics(f, x_curr, u0, h);

        std::vector<double> u0_vec(nu);
        std::vector<double> x_curr_vec(nx);

        for (int i = 0; i < nu; ++i) {
            u0_vec[i] = static_cast<double>(u0(i));
        }
        for (int i = 0; i < nx; ++i) {
            x_curr_vec[i] = static_cast<double>(x_curr(i));
        }
        
        // Store state and control
        state_traj.push_back(std::vector<double>(x_curr_vec));
        control_traj.push_back(std::vector<double>(u0_vec));

        x_curr = x_next;
    }

    // Print results
    std::cout << "\nFinal Trajectory:\n";
    for (std::size_t i = 0; i < state_traj.size(); ++i) {
        std::cout << "Step " << i + 1 << ": x = [";
        for (double val : state_traj[i]) std::cout << val << " ";
        std::cout << "]  u = " << control_traj[i][0] << "\n";
    }

    save_to_csv("state_trajectory.csv", state_traj);
    save_to_csv("control_trajectory.csv", control_traj);

    return 0;
}

