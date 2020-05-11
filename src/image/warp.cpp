#include <cmath>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "warp.h"

const double focal_length = 4000.0 / 15.6 * 16;

void cylindrical_warp_image(cv::Mat& image) {
  cv::Mat warp_image = cv::Mat::zeros(image.size(), image.type());

  int cx = image.cols / 2;
  int cy = image.rows / 2;

  for (int i = 0; i != image.rows; i++) {
    auto row = warp_image.ptr<cv::Vec3b>(i);
    for (int j = 0; j != image.cols; j++) {
      double theta = j - cx;
      double x = std::tan(theta / focal_length) * focal_length;

      double h = i - cy;
      double y = h * std::sqrt(x * x + focal_length * focal_length) / focal_length;

      x += cx;
      int int_x = x;
      double a = x - int_x;

      y += cy;
      int int_y = y;
      double b = y - int_y;

      if (int_x < 0 || int_y < 0 || int_x + 1 >= image.cols || int_y + 1 >= image.rows) continue;

      row[j] = (1 - a) * (1 - b) * image.at<cv::Vec3b>(int_y, int_x) +
               a * (1 - b) * image.at<cv::Vec3b>(int_y, int_x + 1) + a * b * image.at<cv::Vec3b>(int_y + 1, int_x + 1) +
               (1 - a) * b * image.at<cv::Vec3b>(int_y + 1, int_x);
    }
  }

  image = warp_image;
}

void cylindrical_warp_feature_points(std::vector<std::tuple<double, double, double, double>>& feature_points,
                                     const int rows, const int cols) {
  auto warp = [&](auto& i, auto& j) {
    int cx = cols / 2;
    int cy = rows / 2;

    double x = j - cx;
    double theta = focal_length * std::atan(x / focal_length);

    double y = i - cy;
    double h = focal_length * y / std::sqrt(x * x + focal_length * focal_length);

    theta += cx;
    h += cy;

    i = h;
    j = theta;
  };

  for (auto& [i1, j1, i2, j2] : feature_points) {
    warp(i1, j1);
    warp(i2, j2);
  }
}
