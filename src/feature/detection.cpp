#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detection.h"

std::vector<std::tuple<double, double>> calculate_orientation(const cv::Mat& image,
                                                              const std::vector<std::tuple<int, int>>& points) {
  std::cout << "\tcalculate orientation" << std::endl;

  cv::Mat blur;
  cv::GaussianBlur(image, blur, cv::Size(), 1.0, 1.0, cv::BORDER_REPLICATE);

  cv::Mat ix, iy;
  cv::Sobel(blur, ix, -1, 1, 0, 1);
  cv::Sobel(blur, iy, -1, 0, 1, 1);

  std::vector<std::tuple<double, double>> orientation;
  for (const auto& [i, j] : points) {
    auto x = ix.at<double>(i, j);
    auto y = iy.at<double>(i, j);

    auto norm = std::sqrt(x * x + y * y);
    orientation.emplace_back(x / norm, y / norm);
  }

  return orientation;
}

std::vector<std::tuple<double, double>> sub_pixel_refinement(const cv::Mat& strength,
                                                             const std::vector<std::tuple<int, int>>& points) {
  cv::Mat strength_b;
  cv::copyMakeBorder(strength, strength_b, 1, 1, 1, 1, cv::BORDER_REPLICATE);

  std::vector<std::tuple<double, double>> feature_points;
  for (const auto& [i, j] : points) {
    cv::Mat f(strength_b, cv::Rect(j, i, 3, 3));
    auto f_x = (f.at<double>(1, 2) - f.at<double>(1, 0)) / 2;
    auto f_y = (f.at<double>(2, 1) - f.at<double>(0, 1)) / 2;
    auto f_x2 = f.at<double>(1, 2) + f.at<double>(1, 0) - 2 * f.at<double>(1, 1);
    auto f_y2 = f.at<double>(2, 1) + f.at<double>(0, 1) - 2 * f.at<double>(1, 1);
    auto f_xy = (f.at<double>(0, 0) + f.at<double>(2, 2) - f.at<double>(0, 2) - f.at<double>(2, 0)) / 4;

    cv::Mat gradient = (cv::Mat_<double>(2, 1) << f_x, f_y);
    cv::Mat hessian = (cv::Mat_<double>(2, 2) << f_x2, f_xy, f_xy, f_y2);
    cv::Mat x_m = hessian.inv() * gradient;

    feature_points.emplace_back(i + x_m.at<double>(1, 0), j + x_m.at<double>(0, 0));
  }

  return feature_points;
}

std::vector<std::tuple<int, int>> adaptive_non_maximal_supression(std::vector<std::tuple<double, int, int>>& points) {
  std::cout << "\tANMS" << std::endl;

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
  candidates.resize(500);

  std::vector<std::tuple<int, int>> feature_points;
  for (const auto& [s, i, j] : candidates) feature_points.emplace_back(i, j);

  return feature_points;
}

std::vector<std::tuple<double, double, double, double>> Harris_corner_detector(const cv::Mat& image) {
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

  cv::Mat mask;
  cv::dilate(strength, mask, cv::Mat());
  cv::compare(strength, mask, mask, cv::CMP_GE);

  std::vector<std::tuple<double, int, int>> points;
  for (int i = 0; i != strength.rows; i++) {
    auto s = strength.ptr<double>(i);
    auto m = mask.ptr<unsigned char>(i);
    for (int j = 0; j != strength.cols; j++) {
      if (s[j] > 10 && m[j] == 255) {
        points.emplace_back(s[j], i, j);
      }
    }
  }

  auto anms_points = adaptive_non_maximal_supression(points);
  auto spf_points = sub_pixel_refinement(strength, anms_points);

  auto orientation = calculate_orientation(image, anms_points);

  std::vector<std::tuple<double, double, double, double>> feature_points;
  std::transform(spf_points.begin(), spf_points.end(), orientation.begin(), std::back_inserter(feature_points),
                 [](const auto& p, const auto& o) -> std::tuple<double, double, double, double> {
                   auto [i, j] = p;
                   auto [cos, sin] = o;
                   return {i, j, cos, sin};
                 });

  return feature_points;
}

void get_MSOP_features(const cv::Mat& image) {
  std::cout << "[Get MSOP features...]" << std::endl;

  cv::Mat gray;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_64F);

  std::vector<std::vector<std::tuple<double, double, double, double>>> feature_points;
  for (int layer = 0; layer < 5; layer++) {
    std::cout << "\tlayer " << layer << std::endl;

    feature_points.push_back(Harris_corner_detector(gray));

    // cv::Mat test = gray.clone();
    // test.convertTo(test, CV_8U);
    // cv::cvtColor(test, test, cv::COLOR_GRAY2BGR);
    // for (const auto& [i, j, c, s] : feature_points[layer])
    //   cv::circle(test, cv::Point(j, i), 10 / std::pow(2, layer), cv::Scalar(0, 0, 255), 3 / std::pow(2, layer));
    // cv::imwrite("test" + std::to_string(layer) + ".jpg", test);

    cv::pyrDown(gray, gray, cv::Size((gray.cols + 1) / 2, (gray.rows + 1) / 2), cv::BORDER_REPLICATE);
  }
}
