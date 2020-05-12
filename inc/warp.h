#ifndef WARP_
#define WARP_

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

void cylindrical_warp_image(cv::Mat&);
void cylindrical_warp_feature_points(std::vector<std::tuple<float, float, float, float>>&, const int, const int);
void warp_images_together(const std::vector<cv::Mat>&, PanoramasLists&);

#endif
