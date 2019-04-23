#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;
using Eigen::VectorXd;

/**
 * TODO: Set the timestep length and duration
 */
const size_t N = 10;
const double dt = 0.1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
//   simulator around in a circle with a constant steering angle and velocity on
//   a flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
//   presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// Set starting indexes. Will be used by the solver.
const size_t x_i     = 0;
const size_t y_i     = x_i + N;
const size_t psi_i   = y_i + N;
const size_t v_i     = psi_i + N;
const size_t cte_i   = v_i + N;
const size_t epsi_i  = cte_i + N;
const size_t delta_i = epsi_i + N;
const size_t a_i     = delta_i + N - 1;

const double v_ref = 30;

class FG_eval {
 public:
  // Fitted polynomial coefficients
  VectorXd coeffs;
  FG_eval(VectorXd coeffs) { this->coeffs = coeffs; }

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    /**
     * TODO: implement MPC
     * `fg` is a vector of the cost constraints, `vars` is a vector of variable 
     *   values (state & actuators)
     * NOTE: You'll probably go back and forth between this function and
     *   the Solver function below.
     */

    fg[0] = 0; // fg[0] contains the cost.

    // reference state cost error
    for (size_t t = 0; t < N; t++)
    {
        fg[0] += CppAD::pow(vars[cte_i  + t], 2);
        fg[0] += CppAD::pow(vars[epsi_i + t], 2);
        fg[0] += CppAD::pow(vars[v_i    + t] - v_ref, 2);
    }
    
    // control
    for (size_t t = 0; t < N - 1; t++)
    {
        fg[0] += 2 * CppAD::pow(vars[delta_i + t], 2);
        fg[0] += 1 * CppAD::pow(vars[a_i     + t], 2);
    }
    
    // Control change rate
    for (size_t i = 0; i < N - 2; i++)
    {
        fg[0] += 100 * CppAD::pow(vars[delta_i + i + 1] - vars[delta_i + i], 2);
        fg[0] += 100 * CppAD::pow(vars[a_i     + i + 1] - vars[a_i     + i], 2);
    }

    // Initial model constraints
    fg[1 + x_i]    = vars[x_i];
    fg[1 + y_i]    = vars[y_i];
    fg[1 + psi_i]  = vars[psi_i];
    fg[1 + v_i]    = vars[v_i];
    fg[1 + cte_i]  = vars[cte_i];
    fg[1 + epsi_i] = vars[epsi_i];


    for (size_t i = 1; i < N; i++) {
      // states at times t ant t - 1
      AD<double> x_t      = vars[x_i    + i];     // x at time t
      AD<double> x_t_1    = vars[x_i    + i - 1]; // x at time t - 1
      AD<double> y_t      = vars[y_i    + i];     // y at time t
      AD<double> y_t_1    = vars[y_i    + i - 1]; // y at time t - 1
      AD<double> psi_t    = vars[psi_i  + i];     // psi at time t
      AD<double> psi_t_1  = vars[psi_i  + i - 1]; // psi at time t - 1
      AD<double> v_t      = vars[v_i    + i];     // v at time t
      AD<double> v_t_1    = vars[v_i    + i - 1]; // v at time t - 1
      AD<double> cte_t    = vars[cte_i  + i];     // cte at time t
      AD<double> cte_t_1  = vars[cte_i  + i - 1]; // cte at time t - 1
      AD<double> epsi_t   = vars[epsi_i + i];     // epsi at time t
      AD<double> epsi_t_1 = vars[epsi_i + i - 1]; // epsi at time t - 1
      
      // controls
      AD<double> delta_t_1  = vars[delta_i + i - 1];
      AD<double> a_t_1      = vars[a_i     + i - 1];
      AD<double> f_t_1      = coeffs[0] +  \
                              coeffs[1] * x_t_1 +  \
                              coeffs[2] * CppAD::pow(x_t_1, 2)  + \
                              coeffs[3] * CppAD::pow(x_t_1, 3);
      AD<double> psi_t_1_es = CppAD::atan(
                                coeffs[1] +  \
                                2 * coeffs[2] * x_t_1 + \
                                3 * coeffs[3] * CppAD::pow(x_t_1, 2)); 
      
      // set model constrains
      fg[1 + x_i + i]    = x_t    - (x_t_1   + v_t_1 * CppAD::cos(psi_t_1) * dt);
      fg[1 + y_i + i]    = y_t    - (y_t_1   + v_t_1 * CppAD::sin(psi_t_1) * dt);
      fg[1 + psi_i + i]  = psi_t  - (psi_t_1 + (v_t_1 / Lf) * delta_t_1 * dt);
      fg[1 + v_i + i]    = v_t    - (v_t_1   + a_t_1 * dt);
      fg[1 + cte_i + i]  = cte_t  - (f_t_1   - y_t_1      + v_t_1 * CppAD::sin(epsi_t_1) * dt);
      fg[1 + epsi_i + i] = epsi_t - (psi_t_1 - psi_t_1_es + v_t_1 * delta_t_1 / Lf * dt);
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

MPC_Results MPC::Solve(const VectorXd &state, const VectorXd &coeffs) {
  bool ok = true;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  /**
   * TODO: Set the number of model variables (includes both states and inputs).
   * For example: If the state is a 4 element vector, the actuators is a 2
   *   element vector and there are 10 timesteps. The number of variables is:
   *   4 * 10 + 2 * 9
   */
  size_t n_vars = state.size() * N + 2 * (N-1);

  /**
   * TODO: Set the number of constraints
   */
  size_t n_constraints = state.size() * N;

  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (int i = 0; i < n_vars; ++i) {
    vars[i] = 0.0;
  }

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  /**
   * TODO: Set lower and upper limits for variables.
   */

  for (size_t i = 0; i < delta_i; i++) {
    vars_lowerbound[i] = -std::numeric_limits<double>::max();
    vars_upperbound[i] = std::numeric_limits<double>::max();
  }

  for (size_t i = delta_i; i < a_i; i++) {
    vars_lowerbound[i] = -0.436332; // -25 degrees in radians
    vars_upperbound[i] = +0.436332; // +25 degrees in radians
  }

  for (size_t i = a_i; i < n_vars; i++) {
      vars_lowerbound[i] = -1.0;
      vars_upperbound[i] = +1.0;
  }

  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);
  for (int i = 0; i < n_constraints; ++i) {
    constraints_lowerbound[i] = 0.0;
    constraints_upperbound[i] = 0.0;
  }

  constraints_lowerbound[x_i]    = constraints_upperbound[x_i]    = vars[x_i]    = state[0];
  constraints_lowerbound[y_i]    = constraints_upperbound[y_i]    = vars[y_i]    = state[1];
  constraints_lowerbound[psi_i]  = constraints_upperbound[psi_i]  = vars[psi_i]  = state[2];
  constraints_lowerbound[v_i]    = constraints_upperbound[v_i]    = vars[v_i]    = state[3];
  constraints_lowerbound[cte_i]  = constraints_upperbound[cte_i]  = vars[cte_i]  = state[4];
  constraints_lowerbound[epsi_i] = constraints_upperbound[epsi_i] = vars[epsi_i] = state[5];

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  // NOTE: You don't have to worry about these options
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  //   of sparse routines, this makes the computation MUCH FASTER. If you can
  //   uncomment 1 of these and see if it makes a difference or not but if you
  //   uncomment both the computation time should go up in orders of magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost " << cost << std::endl;

  /**
   * TODO: Return the first actuator values. The variables can be accessed with
   *   `solution.x[i]`.
   *
   * {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
   *   creates a 2 element double vector.
   */

  MPC_Results res;
  res.throttle = solution.x[a_i];
  res.steering = solution.x[delta_i] / (0.436332 * Lf); // 0.436332 = 25 degree in rads
  for(size_t i = 0; i < N - 1; i++)
  {
      res.x_vals.push_back(solution.x[x_i + i + 1]);
      res.y_vals.push_back(solution.x[y_i + i + 1]);
  }
  
  return res;
}