#ifndef DESCRIPTION_
#define DESCRIPTION_

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

extern bool fast_patch;

cv::Mat GetDescriptors(const cv::Mat&, const cv::Mat&, const std::vector<std::tuple<float, float>>&);

#endif
