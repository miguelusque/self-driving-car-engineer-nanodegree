#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <uWS/uWS.h>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  // Lanes are numbered (0 | 1 | 2)
  // Start on lane 1 (middle lane)
  int lane = 1;

  // Inicial velocity, and also reference velocity to target.
  double ref_vel = 0.0; // mph

  // True when the ego-car is changing lane.
  bool is_changing_lane = false;
  double end_change_lane_s = 0.0;

  h.onMessage([&map_waypoints_x, &map_waypoints_y, &map_waypoints_s,
               &map_waypoints_dx, &map_waypoints_dy, &lane, &ref_vel,
               &is_changing_lane, &end_change_lane_s]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    const double max_vel = 49.7;
    const double vel_inc = .224;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      auto s = hasData(data);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object

          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];

          // Previous path's end s and d values
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side of the road.
          vector<vector<double>> sensor_fusion = j[1]["sensor_fusion"];

          int prev_size = previous_path_x.size();

          // Avoid collisions - Init
          if (prev_size > 0) {
            car_s = end_path_s;
          }

          // Check if a change lane has finished
          if (is_changing_lane && (car_s >= end_change_lane_s)) {
            is_changing_lane = false;
            end_change_lane_s = 0.0;
          }

          bool too_close = false;
          const double safe_distance = 30.0; // Safe distance between the ego-car and the car ahead of it.
          const double target_x = 30.0; // This is our horizon, 30 meters.

          // Sensor data of any car in a different lanes than ego-car.
          vector<vector<double>> uncurated_data_s = {{}, {}, {}};
          vector<vector<double>> curated_data_s = {};

          // find ref_v to use
          for (int i = 0; i < sensor_fusion.size(); i++) {
            // check if a car is in my lane.
            float d = sensor_fusion[i][6];

            // x and y velocity. Used to calculate the other car speed.
            double vx = sensor_fusion[i][3];
            double vy = sensor_fusion[i][4];
            double check_speed = sqrt((vx * vx) + (vy * vy));

            // s position of the other car.
            double check_car_s = sensor_fusion[i][5];

            // if using previous points, we can project s value outwards in time.
            check_car_s += ((double) prev_size * 0.02 * check_speed);

            if (d > (2 + 4 * lane - 2) && d < (2 + 4 * lane + 2)) {
              // check s values greater than mine and s gap
              // If the other car is in front of us and the distance is less than 'safe distance' meters...
              if ((check_car_s > car_s) && ((check_car_s - car_s) < safe_distance)) {
                // ... doo some logic here, lower reference velocity so we don't crash into the car infront of us,
                // could also flag to try to change lanes.
                too_close = true;
              }
            } else if (!is_changing_lane) {
              // Calculate in which lane the other car is
              int check_car_lane;
              if (d < 4) {
                check_car_lane = 0;
              } else if (d >= 4 && d <= 8) {
                check_car_lane = 1;
              } else {
                check_car_lane = 2;
              }

              // Add sensed-car measures if the sensed-car is at a distance of 1 lane from ego-car
              if (abs(lane - check_car_lane) == 1) {
                uncurated_data_s[check_car_lane].push_back(check_car_s);
              }
            }
          }

          bool change_empty_lane = false;
          int next_lane = -1; // Lane to which we would like to change
          if (too_close && !is_changing_lane) {
            // Decrease velocity ~5 m/s because we are getting close to a car
            ref_vel -= .224;

            // When the ego-car completes a loop, chances are that its s position
            // is much smaller than the sensed-car data.
            // We will ignore that data until the ego-car and the sensed-data
            // are in the same scale (both are at the beginning of the loop.)
            const double prevent_loop = 1000.0;
            // Curate data
            for (int i = 0; i < uncurated_data_s.size(); i++) {
              // Per lane, sort other cars positions
              std::sort(uncurated_data_s[i].begin(), uncurated_data_s[i].end());
              vector<double> lane_data_s = {};
              for (int j = 0; j < uncurated_data_s[i].size(); j++) {
                // If the sensed-car is behind the ego-car
                if (car_s > uncurated_data_s[i][j]) {
                  // Add the sensed-car only if either there is no more sensed data or the next car
                  if (!(j + 1 < uncurated_data_s[i].size() && car_s > uncurated_data_s[i][j + 1])) {
                    lane_data_s.push_back(uncurated_data_s[i][j]);
                  }
                } else if (uncurated_data_s[i][j] - car_s < prevent_loop) {
                  lane_data_s.push_back(uncurated_data_s[i][j]);
                  break;
                }
              }
              curated_data_s.push_back(lane_data_s);

              for (int i = 0; i < curated_data_s.size(); i++) {
                // A lane without any sensed-data should be prefered over any other lane.
                // I will add some 'fake' data to that lane to let the loop above to select
                // that lane over any other lane.
                if (abs(i - lane) == 1 && curated_data_s[i].size() == 0) {
                  // Change to the lane where there is no sensed-cars, from left to right
                  next_lane = i;

                  // Do not try a new lane until current lane change has finished.
                  is_changing_lane = true;

                  // The lane change has finished when car_s >= end_change_lane_s
                  end_change_lane_s = car_s + target_x * 3;
                  change_empty_lane = true;
                  break;
                }
              }
            }

            // Calculate lane to change, if feasible.
            const double safe_gap = 30; // Minimun distance to perform a safe lane change
            double best_gap = 0.0; // Longest gap between our car and others car
            for (int i = 0; !change_empty_lane && i < curated_data_s.size(); i++) {
              for (int j = 0; j < curated_data_s[i].size(); j++) {
                // If the sensed-car is behind the ego-car
                if (car_s > curated_data_s[i][j]) {
                  // Do not consider to change lanes if there is not enough
                  // safe distance with the car behind the ego-car
                  if (car_s - curated_data_s[i][j] < safe_gap) {
                    break; // Do not continue with other sensed car in this lane.
                  } else {
                    // Check if there is a car ahead of the ego-car
                    if (j + 1 < curated_data_s[i].size()) {
                        // Do not consider to change lanes if there is not enough safe distance with the car ahead
                      if (curated_data_s[i][j + 1] - car_s < safe_gap) {
                        break; // Do not continue with other sensed car in this lane.
                      } else if (best_gap < car_s - curated_data_s[i][j + 1]) {
                          // There is enough space between the ego-car and the car ahead.
                          // Because the gap is better than previous gaps, use this lane.
                          best_gap = car_s - curated_data_s[i][j + 1];
                          next_lane = i;
                          break; // Do not continue with other sensed car in this lane.
                      }
                    }  else if (best_gap < car_s - curated_data_s[i][j]) {
                        // There are not remaining cars, and the distance between the sensed-car and the ego-car is >= safe_gap
                        // Therefore, it is safe to change lanes if there are not better options already calculates
                        best_gap = car_s - curated_data_s[i][j];
                        next_lane = i;
                    }
                  }
                } else {
                  // If the sensed-car is ahead of the ego-car
                  if (curated_data_s[i][j] - car_s < safe_gap) {
                  } else if (best_gap < curated_data_s[i][j] - car_s) {
                    // There is enough space between the ego-car and the car ahead.
                    // Because the gap is better than previous gaps, use this lane.
                    best_gap = curated_data_s[i][j] - car_s;
                    next_lane = i;
                    break; // Do not continue with other sensed car in this lane.
                  }
                }
              }
            }

            // If we have found a feasable gap, let's select the next lane to move
            if (best_gap > 0) {
                // Make sure we only change one lane at a time
                if (next_lane > lane) {
                  next_lane = lane + 1;
                } else {
                  next_lane = lane - 1;
                }

                // Do not try a new lane until current lane change has finished.
                is_changing_lane = true;

                // The lane change has finished when car_s >= end_change_lane_s
                end_change_lane_s = car_s + target_x;
            }


            // Check if it is feasible to change lanes
            if (next_lane != -1) {
              lane = next_lane;
            }
          } else if (ref_vel < max_vel) {
            ref_vel += vel_inc; // Increase velocity because we are not close to a car
          }

          // Create a list of widely spaced (x, y) waypoints, evenly spaced at 30m.
          // Later, we will interpolate these waypoints with a spline and fill it in with more points that control speed.
          vector<double> ptsx;
          vector<double> ptsy;

          // Reference x, y, yaw states
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);

          // If previous size is almost empty, use the car as starting reference
          if (prev_size < 2) {
            // Use two points that make the car tangent to the car
            double prev_car_x = car_x - cos(car_yaw);
            double prev_car_y = car_y - sin(car_yaw);

            ptsx.push_back(prev_car_x);
            ptsx.push_back(car_x);

            ptsy.push_back(prev_car_y);
            ptsy.push_back(car_y);
          } else { // Use the previous path's end point as starting reference
            // Redefine reference state as previous path end point
            ref_x = previous_path_x[prev_size - 1];
            ref_y = previous_path_y[prev_size - 1];

            double ref_x_prev = previous_path_x[prev_size - 2];
            double ref_y_prev = previous_path_y[prev_size - 2];
            ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

            // Use two points that make the path tangent to the previous path's end point
            ptsx.push_back(ref_x_prev);
            ptsx.push_back(ref_x);

            ptsy.push_back(ref_y_prev);
            ptsy.push_back(ref_y);
          }

          // In Frenet add evenly 30m spaced points ahead of the starting reference
          vector<double> next_wp0 = getXY(car_s + 30, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp1 = getXY(car_s + 60, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp2 = getXY(car_s + 90, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

          ptsx.push_back(next_wp0[0]);
          ptsx.push_back(next_wp1[0]);
          ptsx.push_back(next_wp2[0]);

          ptsy.push_back(next_wp0[1]);
          ptsy.push_back(next_wp1[1]);
          ptsy.push_back(next_wp2[1]);

          for (int i = 0; i < ptsx.size(); i++) {
            // Shift car reference angle to 0 degrees
            double shift_x = ptsx[i] - ref_x;
            double shift_y = ptsy[i] - ref_y;

            ptsx[i] = shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw);
            ptsy[i] = shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw);
          }

          // Create a spline
          tk::spline s;

          // Set (x, y) points to the spline
          s.set_points(ptsx, ptsy);

          // Define the actual (x, y) points we will use for the planner
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          // Start with all the previous path points from last time
          for (int i = 0; i < previous_path_x.size(); i++) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          // Calculate how to break up spline points so that we travel at our desired reference velocity.
          double target_y = s(target_x);
          double target_dist = sqrt((target_x * target_x) + (target_y * target_y));

          double x_add_on = 0;

          // Fill up the rest of our path planner after filling it with my previous points.
          // Here we will always output 50 points.
          for (int i = 1; i <= 50 - previous_path_x.size(); i++) {
            double N = (target_dist / (0.02 * ref_vel / 2.24)); // 2.24 to change from mph to meters per second.
            double x_point = x_add_on + (target_x) / N;
            double y_point = s(x_point);

            x_add_on = x_point;

            double x_ref = x_point;
            double y_ref = y_point;

            // Rotate back to normal after rotating it earilier
            x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
            y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);

            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }

          json msgJson;

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }

  h.run();
}