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

  bool show_matched_features = false;
  for (int i = 2; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.compare("--show-matched-features") == 0) show_matched_features = true;
  }

  auto image_data = LoadImages(image_dir);
  if (image_data.empty()) {
    std::cerr << "The image directory is empty!" << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<MSOPDescriptor> feature_descriptors;
  for (int index = 0; const auto& image : image_data) {
    std::cout << "[" << index++ << "] ";
    feature_descriptors.push_back(GetMSOPFeatures(image));
  }

  auto match_points = MatchFeatures(feature_descriptors);

  std::cout << "[Warp cylindrical...]" << std::endl;
  std::cout << "\t" << std::flush;
#pragma omp parallel for
  for (size_t i = 0; i < image_data.size(); i++) {
#pragma omp critical
    std::cout << "[" << i << "] " << std::flush;
    for (auto& v : match_points[i]) CylindricalWarpFeaturePoints(v, image_data[i].rows, image_data[i].cols);
    CylindricalWarpImage(image_data[i]);
  }
  std::cout << std::endl;

  auto panoramas_lists = MatchImages(match_points);

  // DrawMatchedFeatures() must before WarpImagesTogether()
  // Since WarpImagesTogether() drops the original image from blending it again
  if (show_matched_features) DrawMatchedFeatures(image_data, panoramas_lists, match_points);

  WarpImagesTogether(image_data, panoramas_lists);

  return EXIT_SUCCESS;
}
