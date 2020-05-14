#ifndef DETECTION_
#define DETECTION_

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

using MSOPDescriptor = std::vector<std::tuple<std::vector<std::tuple<float, float>>, cv::Mat>>;

MSOPDescriptor GetMSOPFeatures(const cv::Mat&);

#endif
