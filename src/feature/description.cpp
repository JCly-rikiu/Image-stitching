#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "description.h"

cv::Mat get_descriptors(const cv::Mat& image, const std::vector<std::tuple<float, float>>& feature_points,
                        const std::vector<float>& orientations) {
  std::cout << " -> descriptor" << std::flush;

  const int patch_size = 40;

  cv::Mat blur;
  cv::GaussianBlur(image, blur, cv::Size(), 1.0, 1.0, cv::BORDER_REPLICATE);

  // Reuse Mat for better cache performance
  cv::Mat rotate;
  cv::Mat patch;
  cv::Mat mean, stddev;

  std::vector<std::tuple<float, float, float>> points_orientations(feature_points.size());
  std::transform(feature_points.begin(), feature_points.end(), orientations.begin(), points_orientations.begin(),
                 [](const auto& p, const auto& o) { return std::tuple_cat(p, std::make_tuple(o)); });

  cv::Mat_<float> descriptors(0, 64);
  for (auto [di, dj, t] : points_orientations) {
    // Sample from downsampled image
    di /= 2;
    dj /= 2;

    // Notice the positive theta and the origin in OpenCV
    cv::warpAffine(blur, rotate, cv::getRotationMatrix2D(cv::Point2f(dj, di), t, 1), blur.size());

    cv::getRectSubPix(rotate, cv::Size(patch_size, patch_size), cv::Point2f(dj, di), patch);
    cv::resize(patch, patch, cv::Size(), 0.2, 0.2);

    cv::meanStdDev(patch, mean, stddev);
    patch = (patch - mean) / stddev;
    cv::patchNaNs(patch);  // Nan may cause problem in cv::flann

    descriptors.push_back(patch.reshape(1, 64).t());
  }

  return descriptors;
}
