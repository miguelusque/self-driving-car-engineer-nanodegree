#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

typedef struct
{
    double throttle;
    double steering;
    std::vector<double> x_vals;
    std::vector<double> y_vals;
} MPC_Results;

class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuations.
  MPC_Results Solve(const Eigen::VectorXd &state, 
                            const Eigen::VectorXd &coeffs);
};

#endif  // MPC_H
