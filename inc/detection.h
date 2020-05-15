#ifndef DETECTION_
#define DETECTION_

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

extern bool fast_anms;

using MSOPDescriptors = std::vector<std::tuple<std::vector<std::tuple<float, float>>, cv::Mat>>;

MSOPDescriptors GetMSOPFeatures(const cv::Mat&);

#endif
