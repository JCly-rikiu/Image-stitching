#ifndef DESCRIPTION_
#define DESCRIPTION_

#include <vector>

#include <opencv2/opencv.hpp>

std::vector<cv::Mat> get_descriptors(const cv::Mat&, const std::vector<std::tuple<int, int>>&,
                                     const std::vector<double>&);

#endif
