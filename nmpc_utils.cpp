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
    const casadi::MX& x_ref_p,
    const casadi::MX& u_ref_p,
    const casadi::MX& Q,
    const casadi::MX& R,
    const casadi::MX& Qf
) {
    casadi::MX cost = 0;
    std::size_t N = us.size();

    for (std::size_t k = 0; k < N; ++k) {
        casadi::MX xk = xs[k];
        casadi::MX uk = us[k];
        casadi::MX x_ref_k = x_ref_p(casadi::Slice(), k);
        casadi::MX dx = xk - x_ref_k;
        casadi::MX u_ref_k = u_ref_p(casadi::Slice(), k);
        casadi::MX du = uk - u_ref_k;
        cost += casadi::MX::mtimes({dx.T(), Q, dx}) + casadi::MX::mtimes({du.T(), R, du});
    }
    casadi::MX dx_terminal = xs[N] - x_ref_p(casadi::Slice(), N);
    cost += casadi::MX::mtimes({dx_terminal.T(), Qf, dx_terminal});
    return cost;
}

casadi::MX create_dynamics_constraints(
    const std::vector<casadi::MX>& xs,
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

casadi::DM simulate_dynamics(const casadi::Function& f, const casadi::DM& x, const casadi::DM& u, double h) {
    casadi::DM k1 = f(casadi::DMVector{x, u})[0];
    casadi::DM k2 = f(casadi::DMVector{x + 0.5 * h * k1, u})[0];
    casadi::DM k3 = f(casadi::DMVector{x + 0.5 * h * k2, u})[0];
    casadi::DM k4 = f(casadi::DMVector{x + h * k3, u})[0];
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4);
}

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

NMPCProblem::NMPCProblem(int N_, int nx_, int nu_, double h,
                        const casadi::Function& f,
                        const casadi::DM& Q, const casadi::DM& R, const casadi::DM& Qf)
    : N(N_), nx(nx_), nu(nu_)
{
    std::vector<casadi::MX> xs, us;
    for (int k = 0; k <= N; ++k)
        xs.push_back(casadi::MX::sym("x" + std::to_string(k), nx));
    for (int k = 0; k < N; ++k)
        us.push_back(casadi::MX::sym("u" + std::to_string(k), nu));

    casadi::MX x_ref_p = casadi::MX::sym("x_ref_p", nx, N+1);
    casadi::MX u_ref_p = casadi::MX::sym("u_ref_p", nu, N);

    casadi::MX cost = create_cost_function(xs, us, x_ref_p, u_ref_p, Q, R, Qf);
    casadi::MX g = create_dynamics_constraints(xs, us, f, h);

    casadi::MX opt_vars = casadi::MX::vertcat(xs);
    opt_vars = casadi::MX::vertcat({opt_vars, casadi::MX::vertcat(us)});
    casadi::MX parameters = casadi::MX::vertcat({casadi::MX::vec(x_ref_p), casadi::MX::vec(u_ref_p)});

    casadi::MXDict nlp;
    nlp["x"] = opt_vars;
    nlp["p"] = parameters;
    nlp["f"] = cost;
    nlp["g"] = g;

    casadi::Dict opts;
    opts["print_time"] = false;
    opts["verbose"] = false;
    opts["max_iter"] = 5000;
    solver = casadi::nlpsol("solver", "qrsqp", nlp, opts);
}

casadi::DM NMPCProblem::solve(const casadi::DM& x_init,
                            const casadi::DM& x_ref,
                            const casadi::DM& u_ref,
                            const casadi::DM& u_min,
                            const casadi::DM& u_max,
                            const casadi::DM& x_min,
                            const casadi::DM& x_max,
                            const casadi::DM* warm_start_x0)
{
    int n_vars = (N+1)*nx + N*nu;
    int n_cons = N*nx;

    std::vector<double> x0(n_vars, 0.0);
    if (warm_start_x0 && warm_start_x0->size1() == n_vars) {
        for (int i = 0; i < n_vars; i++) {
            x0[i] = static_cast<double>((*warm_start_x0)(i));
        }
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
        lbx[j] = static_cast<double>(x_init(j));
        ubx[j] = static_cast<double>(x_init(j));
    }

    casadi::DM p_values = casadi::DM::vertcat({casadi::DM::vec(x_ref), casadi::DM::vec(u_ref)});

    casadi::DMDict arg;
    arg["x0"] = x0;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    arg["lbx"] = lbx;
    arg["ubx"] = ubx;
    arg["p"] = p_values;

    casadi::DMDict result = solver(arg);
    return result.at("x");
}
