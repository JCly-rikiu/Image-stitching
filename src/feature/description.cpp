#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "description.h"

std::vector<cv::Mat> get_descriptors(const cv::Mat& image, const std::vector<std::tuple<int, int>>& feature_points,
                                     const std::vector<double>& orientations) {
  std::cout << "\tget descriptor" << std::endl;

  const int patch_half_size = 20;
  const int patch_size = patch_half_size * 2;

  cv::Mat blur;
  cv::GaussianBlur(image, blur, cv::Size(), 1.0, 1.0, cv::BORDER_REPLICATE);
  // Make border for ROI out of bound issue
  cv::copyMakeBorder(blur, blur, patch_half_size, patch_half_size, patch_half_size, patch_half_size,
                     cv::BORDER_REPLICATE);

  // Reuse Mat for better cache performance
  cv::Mat rotate;
  cv::Mat patch;
  cv::Mat mean, stddev;

  std::vector<std::tuple<int, int, double>> points_orientations(feature_points.size());
  std::transform(feature_points.begin(), feature_points.end(), orientations.begin(), points_orientations.begin(),
                 [](const auto& p, const auto& o) { return std::tuple_cat(p, std::make_tuple(o)); });

  std::vector<cv::Mat> descriptors;
  for (auto [i, j, t] : points_orientations) {
    i = i / 2;
    j = j / 2;

    // Notice the positive theta and the origin in OpenCV
    cv::warpAffine(blur, rotate, cv::getRotationMatrix2D(cv::Point(j + patch_half_size, i + patch_half_size), t, 1),
                   blur.size());

    patch = rotate(cv::Rect(j, i, patch_size, patch_size)).clone();
    cv::resize(patch, patch, cv::Size(), 0.2, 0.2);

    cv::meanStdDev(patch, mean, stddev);
    patch = (patch - mean) / stddev;

    descriptors.push_back(patch);
  }

  return descriptors;
}
