#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "description.h"
#include "detection.h"

const double radian_to_degree = 180 / std::acos(-1);

std::vector<double> calculate_orientation(const cv::Mat& image, const std::vector<std::tuple<double, double>>& points) {
  std::cout << "\tcalculate orientation" << std::endl;

  cv::Mat blur;
  cv::GaussianBlur(image, blur, cv::Size(), 4.5, 4.5, cv::BORDER_REPLICATE);

  cv::Mat ix, iy;
  cv::Sobel(blur, ix, -1, 1, 0, 1);
  cv::Sobel(blur, iy, -1, 0, 1, 1);

  std::vector<double> orientation;
  for (const auto& [di, dj] : points) {
    int i = static_cast<int>(std::nearbyint(di));
    int j = static_cast<int>(std::nearbyint(dj));

    double x = ix.at<double>(i, j);
    double y = iy.at<double>(i, j);

    double norm = std::sqrt(x * x + y * y);
    orientation.push_back(std::acos(x / norm) * radian_to_degree);
  }

  return orientation;
}

std::vector<std::tuple<double, double>> sub_pixel_refinement(const cv::Mat& strength,
                                                             const std::vector<std::tuple<int, int>>& points) {
  std::cout << "\tsub pixel refinement" << std::endl;

  cv::Mat strength_b;
  // Make border for ROI out of bound issue
  cv::copyMakeBorder(strength, strength_b, 1, 1, 1, 1, cv::BORDER_REPLICATE);

  std::vector<std::tuple<double, double>> feature_points;
  for (const auto& [i, j] : points) {
    cv::Mat f(strength_b, cv::Rect(j, i, 3, 3));
    double f_x = (f.at<double>(1, 2) - f.at<double>(1, 0)) / 2;
    double f_y = (f.at<double>(2, 1) - f.at<double>(0, 1)) / 2;
    double f_x2 = f.at<double>(1, 2) + f.at<double>(1, 0) - 2 * f.at<double>(1, 1);
    double f_y2 = f.at<double>(2, 1) + f.at<double>(0, 1) - 2 * f.at<double>(1, 1);
    double f_xy = (f.at<double>(0, 0) + f.at<double>(2, 2) - f.at<double>(0, 2) - f.at<double>(2, 0)) / 4;

    cv::Mat gradient = (cv::Mat_<double>(2, 1) << f_x, f_y);
    cv::Mat hessian = (cv::Mat_<double>(2, 2) << f_x2, f_xy, f_xy, f_y2);
    cv::Mat x_m = hessian.inv() * gradient;

    feature_points.emplace_back(i + x_m.at<double>(1, 0), j + x_m.at<double>(0, 0));
  }

  return feature_points;
}

std::vector<std::tuple<int, int>> adaptive_non_maximal_supression(std::vector<std::tuple<double, int, int>>& points) {
  std::cout << "\tANMS" << std::endl;

  const int feature_number = 300;

  std::sort(points.begin(), points.end(), std::greater<std::tuple<double, int, int>>());

  std::vector<std::tuple<int, int, int>> candidates;
  for (auto current = points.begin(); current != points.end(); current++) {
    const auto& [current_s, current_i, current_j] = *current;

    auto min_radius = std::numeric_limits<int>::max();
    for (auto previous = points.begin(); previous != current; previous++) {
      const auto& [previous_s, previous_i, previous_j] = *previous;

      auto radius =
          (current_i - previous_i) * (current_i - previous_i) + (current_j - previous_j) * (current_j - previous_j);
      min_radius = std::min(min_radius, radius);
    }

    candidates.emplace_back(min_radius, current_i, current_j);
  }

  std::sort(candidates.begin(), candidates.end(), std::greater<std::tuple<int, int, int>>());
  if (candidates.size() > feature_number) candidates.resize(feature_number);

  std::vector<std::tuple<int, int>> feature_points;
  for (const auto& [s, i, j] : candidates) feature_points.emplace_back(i, j);

  return feature_points;
}

std::tuple<std::vector<std::tuple<double, double>>, std::vector<double>> Harris_corner_detector(const cv::Mat& image) {
  std::cout << "\tHarris Corner Detector" << std::endl;

  cv::Mat blur;
  cv::GaussianBlur(image, blur, cv::Size(), 1.0, 1.0, cv::BORDER_REPLICATE);

  cv::Mat ix, iy;
  cv::Sobel(blur, ix, -1, 1, 0, 1);
  cv::Sobel(blur, iy, -1, 0, 1, 1);

  cv::Mat ixx, iyy, ixy;
  cv::multiply(ix, ix, ixx);
  cv::multiply(iy, iy, iyy);
  cv::multiply(ix, iy, ixy);

  cv::GaussianBlur(ixx, ixx, cv::Size(), 1.5, 1.5, cv::BORDER_REPLICATE);
  cv::GaussianBlur(iyy, iyy, cv::Size(), 1.5, 1.5, cv::BORDER_REPLICATE);
  cv::GaussianBlur(ixy, ixy, cv::Size(), 1.5, 1.5, cv::BORDER_REPLICATE);

  cv::Mat det = ixx.mul(iyy) - ixy.mul(ixy);
  cv::Mat trace = ixx + iyy;
  cv::Mat strength = det / trace;

  // Find local maxima
  cv::Mat mask;
  cv::dilate(strength, mask, cv::Mat());
  cv::compare(strength, mask, mask, cv::CMP_GE);

  std::vector<std::tuple<double, int, int>> points;
  for (int i = 0; i != strength.rows; i++) {
    auto s = strength.ptr<double>(i);
    auto m = mask.ptr<unsigned char>(i);
    for (int j = 0; j != strength.cols; j++)
      if (s[j] > 10 && m[j] == 255) points.emplace_back(s[j], i, j);
  }

  auto anms_points = adaptive_non_maximal_supression(points);
  auto spf_points = sub_pixel_refinement(strength, anms_points);

  auto orientations = calculate_orientation(image, spf_points);

  return {spf_points, orientations};
}

MSOPDescriptor get_MSOP_features(const cv::Mat& image) {
  std::cout << "[Get MSOP features...]" << std::endl;

  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_64F);

  MSOPDescriptor feature_descriptors;
  for (int layer = 0; layer < 3; layer++) {
    std::cout << "\t[layer " << layer << "]" << std::endl;

    auto [feature_points, orientations] = Harris_corner_detector(gray);

    cv::pyrDown(gray, gray, cv::Size(gray.cols / 2, gray.rows / 2), cv::BORDER_REPLICATE);

    auto descriptors = get_descriptors(gray, feature_points, orientations);

    for (auto& [i, j] : feature_points) {
      i *= std::pow(2, layer);
      j *= std::pow(2, layer);
    }

    feature_descriptors.emplace_back(feature_points, descriptors);
  }

  return feature_descriptors;
}
