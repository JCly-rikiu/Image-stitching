#ifndef DETECTION_
#define DETECTION_

#include <vector>

#include <opencv2/opencv.hpp>

std::vector<std::tuple<std::vector<std::tuple<int, int>>, std::vector<cv::Mat>>> get_MSOP_features(const cv::Mat&);

#endif
