#ifndef DETECTION_
#define DETECTION_

#include <vector>

#include <opencv2/opencv.hpp>

typedef std::vector<std::tuple<std::vector<std::tuple<double, double>>, std::vector<cv::Mat>>> MSOPDescriptor;

MSOPDescriptor get_MSOP_features(const cv::Mat&);

#endif
