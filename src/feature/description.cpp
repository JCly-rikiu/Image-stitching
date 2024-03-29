#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "description.h"

bool fast_patch = false;

std::vector<float> CalculateOrientation(const cv::Mat& image, const std::vector<std::tuple<float, float>>& points) {
  std::vector<float> orientation;

  if (fast_patch) {
    orientation.resize(points.size());
    return orientation;
  }

  std::cout << " -> orientation" << std::flush;

  const float radian_to_degree = 180 / std::acos(-1);

  cv::Mat blur;
  cv::GaussianBlur(image, blur, cv::Size(), 4.5, 4.5, cv::BORDER_REPLICATE);

  cv::Mat ix, iy;
  cv::Sobel(blur, ix, -1, 1, 0, 1);
  cv::Sobel(blur, iy, -1, 0, 1, 1);

  for (const auto [di, dj] : points) {
    int i = static_cast<int>(std::nearbyint(di));
    int j = static_cast<int>(std::nearbyint(dj));

    if (i < 0)
      i = 0;
    else if (i >= blur.rows)
      i = blur.rows - 1;
    if (j < 0)
      j = 0;
    else if (j >= blur.cols)
      j = blur.cols - 1;

    float x = ix.at<float>(i, j);
    float y = iy.at<float>(i, j);

    float norm = std::sqrt(x * x + y * y);
    orientation.push_back(std::acos(x / norm) * radian_to_degree);
  }

  return orientation;
}

cv::Mat GetDescriptors(const cv::Mat& image, const cv::Mat& image_down,
                       const std::vector<std::tuple<float, float>>& feature_points) {
  auto orientations = CalculateOrientation(image, feature_points);

  std::cout << " -> descriptor" << std::flush;

  const int patch_size = 40;
  const int small_patch_size = 8;
  const int feature_dimension = small_patch_size * small_patch_size;

  cv::Mat blur;
  cv::GaussianBlur(image_down, blur, cv::Size(), 1.0, 1.0, cv::BORDER_REPLICATE);

  std::vector<std::tuple<float, float, float>> points_orientations(feature_points.size());
  std::transform(feature_points.begin(), feature_points.end(), orientations.begin(), points_orientations.begin(),
                 [](const auto& p, const auto& o) { return std::tuple_cat(p, std::make_tuple(o)); });

  cv::Mat_<float> descriptors(points_orientations.size(), feature_dimension);
#pragma omp parallel for
  for (auto it = points_orientations.begin(); it < points_orientations.end(); it++) {
    auto [di, dj, t] = *it;

    // Sample from downsampled image
    di /= 2;
    dj /= 2;

    cv::Mat rotate;
    if (fast_patch) {
      rotate = blur;
    } else {
      // Notice the positive theta and the origin in OpenCV
      cv::warpAffine(blur, rotate, cv::getRotationMatrix2D(cv::Point(dj, di), t, 1), blur.size());
    }

    cv::Mat patch;
    cv::getRectSubPix(rotate, cv::Size(patch_size, patch_size), cv::Point(dj, di), patch);
    cv::resize(patch, patch, cv::Size(small_patch_size, small_patch_size));

    cv::Mat mean, stddev;
    cv::meanStdDev(patch, mean, stddev);
    patch = (patch - mean) / stddev;
    cv::patchNaNs(patch);  // NaN may cause problem in cv::flann

    descriptors.row(it - points_orientations.begin()) = patch.reshape(1, feature_dimension).t();
  }

  std::cout << std::endl;

  return descriptors;
}
