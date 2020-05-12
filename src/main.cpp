#include <iostream>
#include <string>

#include "detection.h"
#include "image.h"
#include "match.h"
#include "warp.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Please provide the image directory!" << std::endl;
    return EXIT_FAILURE;
  }
  std::string image_dir = argv[1];
  if (image_dir.back() != '/') image_dir += "/";

  auto image_data = load_images(image_dir);
  if (image_data.empty()) {
    std::cerr << "The image directory is empty!" << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<MSOPDescriptor> feature_descriptors;
  for (const auto& image : image_data) feature_descriptors.push_back(get_MSOP_features(image));

  auto match_points = match_features(feature_descriptors);

  std::cout << "[Warp cylindrical...]" << std::endl;
  for (size_t i = 0; i != image_data.size(); i++) {
    for (auto& v : match_points[i])
      cylindrical_warp_feature_points(v, image_data[i].rows, image_data[i].cols);
    cylindrical_warp_image(image_data[i]);
  }

  auto panoramas_lists = match_images(match_points);

  warp_images_together(image_data, panoramas_lists);

  // cv::Mat test;
  // cv::hconcat(image_data[0], image_data[1], test);

  // for (auto [i1, j1, i2, j2] : match_points[0][1]) {
  //   const int x_shift = image_data[0].cols;

  //   cv::circle(test, cv::Point2d(j1, i1), 10, cv::Scalar(0, 0, 255), 3);
  //   cv::circle(test, cv::Point2d(j2 + x_shift, i2), 10, cv::Scalar(0, 0, 255), 3);

  //   cv::line(test, cv::Point2d(j1, i1), cv::Point2d(j2 + x_shift, i2), cv::Scalar(0, 255, 0), 2);
  // }

  // cv::imwrite("test.jpg", test);

  return EXIT_SUCCESS;
}
