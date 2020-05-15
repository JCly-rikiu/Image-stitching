#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "description.h"
#include "detection.h"

std::vector<std::tuple<float, float>> SubPixelRefinement(const cv::Mat& strength,
                                                         const std::vector<std::tuple<int, int>>& points) {
  std::cout << " -> subpixel refinement" << std::flush;

  std::vector<std::tuple<float, float>> feature_points;
  for (const auto [i, j] : points) {
    cv::Mat f;
    cv::getRectSubPix(strength, cv::Size(3, 3), cv::Point2f(j, i), f);

    float f_x = (f.at<float>(1, 2) - f.at<float>(1, 0)) / 2;
    float f_y = (f.at<float>(2, 1) - f.at<float>(0, 1)) / 2;
    float f_x2 = f.at<float>(1, 2) + f.at<float>(1, 0) - 2 * f.at<float>(1, 1);
    float f_y2 = f.at<float>(2, 1) + f.at<float>(0, 1) - 2 * f.at<float>(1, 1);
    float f_xy = (f.at<float>(0, 0) + f.at<float>(2, 2) - f.at<float>(0, 2) - f.at<float>(2, 0)) / 4;

    cv::Mat gradient = (cv::Mat_<float>(2, 1) << f_x, f_y);
    cv::Mat hessian = (cv::Mat_<float>(2, 2) << f_x2, f_xy, f_xy, f_y2);
    cv::Mat x_m = hessian.inv() * gradient;

    feature_points.emplace_back(i + x_m.at<float>(1, 0), j + x_m.at<float>(0, 0));
  }

  return feature_points;
}

std::vector<std::tuple<int, int>> AdaptiveNonMaximalSuppression(std::vector<std::tuple<float, int, int>>& points) {
  std::cout << " -> ANMS " << std::flush;

  const int feature_number = 500;

  std::sort(points.begin(), points.end(), std::greater<std::tuple<float, int, int>>());

  std::vector<std::tuple<int, int, int>> candidates(points.size());
#pragma omp parallel for
  for (auto current = points.begin(); current < points.end(); current++) {
    auto [current_s, current_i, current_j] = *current;

    auto min_radius = std::numeric_limits<int>::max();
    for (auto previous = points.begin(); previous != current; previous++) {
      auto [previous_s, previous_i, previous_j] = *previous;

      auto radius =
          (current_i - previous_i) * (current_i - previous_i) + (current_j - previous_j) * (current_j - previous_j);
      min_radius = std::min(min_radius, radius);
    }

    candidates[current - points.begin()] = {min_radius, current_i, current_j};
  }

  std::sort(candidates.begin(), candidates.end(), std::greater<std::tuple<int, int, int>>());
  if (candidates.size() > feature_number) candidates.resize(feature_number);

  std::vector<std::tuple<int, int>> feature_points;
  for (const auto [s, i, j] : candidates) feature_points.emplace_back(i, j);
  std::cout << "raidus: " << std::get<0>(candidates.back()) << std::flush;

  return feature_points;
}

std::vector<std::tuple<float, float>> HarrisCornerDetector(const cv::Mat& image) {
  std::cout << " detector" << std::flush;

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

  std::vector<std::tuple<float, int, int>> points;
  for (int i = 0; i != strength.rows; i++) {
    auto s = strength.ptr<float>(i);
    auto m = mask.ptr<unsigned char>(i);
    for (int j = 0; j != strength.cols; j++)
      if (s[j] > 10 && m[j] == 255) points.emplace_back(s[j], i, j);
  }

  auto anms_points = AdaptiveNonMaximalSuppression(points);
  auto spf_points = SubPixelRefinement(strength, anms_points);

  return spf_points;
}

MSOPDescriptors GetMSOPFeatures(const cv::Mat& image) {
  std::cout << "[Get MSOP features...]" << std::endl;

  cv::Mat gray, gray_down;
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  gray.convertTo(gray, CV_32F);

  MSOPDescriptors feature_descriptors;
  for (int layer = 0; layer < 5; layer++) {
    // If the image has more then 1M pixels, skip layer 0
    if (layer == 0 && image.rows * image.cols > 1'000'000) {
      std::cout << "\t[layer " << layer << "] skipped (image has more than 1M pixels)" << std::endl;
      cv::pyrDown(gray, gray, cv::Size(gray.cols / 2, gray.rows / 2), cv::BORDER_REPLICATE);
      continue;
    }

    std::cout << "\t[layer " << layer << "]" << std::flush;

    auto feature_points = HarrisCornerDetector(gray);

    cv::pyrDown(gray, gray_down, cv::Size(gray.cols / 2, gray.rows / 2), cv::BORDER_REPLICATE);

    auto descriptors = GetDescriptors(gray, gray_down, feature_points);

    for (auto& [i, j] : feature_points) {
      i *= std::pow(2, layer);
      j *= std::pow(2, layer);
    }

    feature_descriptors.emplace_back(feature_points, descriptors);

    gray = gray_down;
  }

  return feature_descriptors;
}
