#include "kalman_filter.h"
#include <iostream>


using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  // Please notice this part is based on code from Lesson 5, Part 14
  std::cout  << "KalmanFilter::Predict()" << std::endl;
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  // Please notice this part is based on code from Lesson 5, Part 14
  std::cout << "KalmanFilter::Update" << std::endl;
 	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;

  // Update the remaining values.
  // This is common for Lidar and Radar
  UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  std::cout << "KalmanFilter::UpdateEKF" << std::endl;


	//recover state parameters
	float px = x_(0);
	float py = x_(1);

  // Calculate ro and avoid division by zero
  float ro = sqrt(px*px + py*py);
  if(fabs(ro) < 0.0001) {
    return;
  }

  // Calculate theta avoiding atan2(0,0)
  if(px == 0 && py == 0) {
    return;
  }
  float theta = atan2(py, px);

  // Calculate ro_dot
  float vx = x_(2);
	float vy = x_(3);
  float ro_dot = (px*vx + py*vy)/ro;

  VectorXd z_pred(3);
  z_pred << ro, theta, ro_dot;

  VectorXd y = z - z_pred;

  // Normalizing Angles between -pi and pi
  y[1] = atan2(sin(y[1]), cos(y[1]));


  // Update the remaining values.
  // This is common for Lidar and Radar
  UpdateCommon(y);
}

void KalmanFilter::UpdateCommon(const VectorXd &y) {
	MatrixXd Ht = H_.transpose();
	MatrixXd PHt = P_ * Ht;
  MatrixXd S = H_ * PHt + R_;
	MatrixXd Si = S.inverse();
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}