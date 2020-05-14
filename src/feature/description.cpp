#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

#include <omp.h>

#include <opencv2/opencv.hpp>

#include "description.h"

cv::Mat GetDescriptors(const cv::Mat& image, const std::vector<std::tuple<float, float>>& feature_points,
                       const std::vector<float>& orientations) {
  std::cout << " -> descriptor" << std::flush;

  const int patch_size = 40;
  const int small_patch_size = 8;
  const int feature_dimension = small_patch_size * small_patch_size;

  cv::Mat blur;
  cv::GaussianBlur(image, blur, cv::Size(), 1.0, 1.0, cv::BORDER_REPLICATE);

  std::vector<std::tuple<float, float, float>> points_orientations(feature_points.size());
  std::transform(feature_points.begin(), feature_points.end(), orientations.begin(), points_orientations.begin(),
                 [](const auto& p, const auto& o) { return std::tuple_cat(p, std::make_tuple(o)); });

  std::vector<cv::Mat> patch_pool(omp_get_max_threads());

  cv::Mat_<float> descriptors(points_orientations.size(), feature_dimension);
#pragma omp parallel for
  for (auto it = points_orientations.begin(); it < points_orientations.end(); it++) {
    auto [di, dj, t] = *it;

    // Sample from downsampled image
    di /= 2;
    dj /= 2;

    // Notice the positive theta and the origin in OpenCV
    cv::Mat rotate;
    cv::warpAffine(blur, rotate, cv::getRotationMatrix2D(cv::Point2f(dj, di), t, 1), blur.size());

    int tid = omp_get_thread_num();
    cv::Mat& patch = patch_pool[tid];
    cv::getRectSubPix(rotate, cv::Size(patch_size, patch_size), cv::Point2f(dj, di), patch);
    cv::resize(patch, patch, cv::Size(small_patch_size, small_patch_size));

    cv::Mat mean, stddev;
    cv::meanStdDev(patch, mean, stddev);
    patch = (patch - mean) / stddev;
    cv::patchNaNs(patch);  // NaN may cause problem in cv::flann

    descriptors.row(it - points_orientations.begin()) = patch.reshape(1, feature_dimension).t();
  }

  return descriptors;
}
