#ifndef DESCRIPTION_
#define DESCRIPTION_

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

std::vector<cv::Mat> get_descriptors(const cv::Mat&, const std::vector<std::tuple<double, double>>&,
                                     const std::vector<double>&);

#endif
