#ifndef DESCRIPTION_
#define DESCRIPTION_

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

cv::Mat GetDescriptors(const cv::Mat&, const std::vector<std::tuple<float, float>>&, const std::vector<float>&);

#endif
