#ifndef WARP_
#define WARP_

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "match.h"

void CylindricalWarpImage(cv::Mat&);
void CylindricalWarpFeaturePoints(std::vector<std::tuple<float, float, float, float>>&, const int, const int);
void WarpImagesTogether(const std::vector<cv::Mat>&, PanoramaLists&);
void DrawFeatureMatches(const std::vector<cv::Mat>&, const PanoramaLists&, const FeatureMatches&);

#endif
