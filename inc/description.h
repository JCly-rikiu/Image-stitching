#ifndef DESCRIPTION_
#define DESCRIPTION_

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

cv::Mat get_descriptors(const cv::Mat&, const std::vector<std::tuple<float, float>>&, const std::vector<float>&);

#endif
