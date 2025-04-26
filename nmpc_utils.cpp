#include "nmpc_utils.hpp"
#include <fstream>
#include <iostream>

casadi::MX rk4(const casadi::Function& f, casadi::MX x, casadi::MX u, double h) {
    casadi::MX k1 = f(casadi::MXVector{x, u})[0];
    casadi::MX k2 = f(casadi::MXVector{x + 0.5 * h * k1, u})[0];
    casadi::MX k3 = f(casadi::MXVector{x + 0.5 * h * k2, u})[0];
    casadi::MX k4 = f(casadi::MXVector{x + h * k3, u})[0];
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);    
}

casadi::MX create_cost_function(
    const std::vector<casadi::MX>& xs,
    const std::vector<casadi::MX>& us,
    const casadi::MX& x_ref,
    const casadi::MX& u_ref,
    const casadi::MX& Q,
    const casadi::MX& R,
    const casadi::MX& Qf    // Terminal weight
) {
    casadi::MX cost =0;
    std::size_t N = us.size();

    for (std::size_t k = 0; k < N; ++k) {
        casadi::MX xk = xs[k];
        casadi::MX uk = us[k];

        casadi::MX x_ref_k = x_ref(casadi::Slice(), k);
        casadi::MX dx = xk - x_ref_k;
        casadi::MX du = uk - u_ref;
        cost += casadi::MX::mtimes({dx.T(), Q, dx}) + casadi::MX::mtimes({du.T(), R, du});
        }
    casadi::MX dx_terminal = xs[N] - x_ref(casadi::Slice(), N);
    cost += casadi::MX::mtimes({dx_terminal.T(), Qf, dx_terminal});

    return cost;
}

casadi::MX create_dynamics_constraints(const std::vector<casadi::MX>& xs,
                                    const std::vector<casadi::MX>& us,
                                    const casadi::Function& f,
                                    double h) {
    std::vector<casadi::MX> constraints;
    int N = us.size();
    for (int k = 0; k < N; ++k) {
        constraints.push_back(xs[k + 1] - rk4(f, xs[k], us[k], h));
    }
    return casadi::MX::vertcat(constraints);
}

casadi::DM solve_qp(
    int N,
    double h,
    int nx,
    int nu,
    const casadi::Function& f,
    std::vector<double> x_init,
    const casadi::DM& x_ref,                       
    const casadi::DM& u_ref,
    const casadi::DM& Q,
    const casadi::DM& R,
    const casadi::DM& Qf,
    const casadi::DM& u_min,
    const casadi::DM& u_max,
    const casadi::DM& x_min,
    const casadi::DM& x_max,
    const std::vector<double>* warm_start_x0
) {
        // Symbolic variables
    std::vector<casadi::MX> xs, us;
    for (int k = 0; k <= N; ++k)
        xs.push_back(casadi::MX::sym("x" + std::to_string(k), nx));
    for (int k = 0; k < N; ++k)
        us.push_back(casadi::MX::sym("u" + std::to_string(k), nu));

    casadi::MX cost = create_cost_function(xs, us, x_ref, u_ref, Q, R, Qf);
    casadi::MX g = create_dynamics_constraints(xs, us, f, h);

    // Stack decision variables to get the optimizable variables
    casadi::MX opt_vars = casadi::MX::vertcat(xs);          
    opt_vars = casadi::MX::vertcat({opt_vars, casadi::MX::vertcat(us)});        

    casadi::MXDict nlp;
    nlp["x"] = opt_vars;
    nlp["f"] = cost;
    nlp["g"] = g;                          

    // Bounds and initial guess
    int n_vars = opt_vars.size1();
    int n_cons = g.size1();

    std::vector<double> x0(n_vars, 0.0);
    if (warm_start_x0 && warm_start_x0->size() == n_vars) {
        x0 = *warm_start_x0;
    }    
    std::vector<double> lbg(n_cons, 0.0);
    std::vector<double> ubg(n_cons, 0.0);
    std::vector<double> lbx(n_vars, -casadi::inf);
    std::vector<double> ubx(n_vars,  casadi::inf);

    // State bounds
    for (int k = 0; k <= N; ++k) {
        int idx = k * nx;
        for (int j = 0; j < nx; ++j) {
            lbx[idx + j] = static_cast<double>(x_min(j));
            ubx[idx + j] = static_cast<double>(x_max(j));
        }
    }
    
    // Input bounds
    for (int k = 0; k < N; ++k) {
        int idx = (N + 1) * nx + k * nu;
        for (int j = 0; j < nu; ++j) {
            lbx[idx + j] = static_cast<double>(u_min(j));
            ubx[idx + j] = static_cast<double>(u_max(j));
        }
    }
    
    
    // Fix initial state
    for (int j = 0; j < nx; ++j) {
        lbx[j] = x_init[j];
        ubx[j] = x_init[j];
    }
    
    // Solver
    casadi::Dict opts;
    opts["print_time"] = false;
    opts["verbose"] = false;
    opts["max_iter"] = 5000;
    casadi::Function solver = casadi::nlpsol("solver", "qrsqp", nlp, opts);

    casadi::DMDict arg;
    arg["x0"] = x0;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    arg["lbx"] = lbx;
    arg["ubx"] = ubx;

    casadi::DMDict result = solver(arg);
    return result.at("x");
}

//------------- Simulate dynamics using RK4 (generates actual values)---------------//
casadi::DM simulate_dynamics(const casadi::Function& f, const casadi::DM& x, const casadi::DM& u, double h) {
    casadi::DM k1 = f(casadi::DMVector{x, u})[0];
    casadi::DM k2 = f(casadi::DMVector{x + 0.5 * h * k1, u})[0];
    casadi::DM k3 = f(casadi::DMVector{x + 0.5 * h * k2, u})[0];
    casadi::DM k4 = f(casadi::DMVector{x + h * k3, u})[0];

    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
}

//----------- Save results to CSV --------------//
void save_to_csv(const std::string& filename, const std::vector<std::vector<double>>& data) {

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    for (const auto& row : data) {
        for (std::size_t j = 0; j < row.size(); ++j) {
            file << row[j];
            if (j < row.size() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
}