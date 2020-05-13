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

  auto image_data = load_images(image_dir);
  if (image_data.empty()) {
    std::cerr << "The image directory is empty!" << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<MSOPDescriptor> feature_descriptors;
  for (int index = 0; const auto& image : image_data) {
    std::cout << "[" << index++ << "] ";
    feature_descriptors.push_back(get_MSOP_features(image));
  }

  auto match_points = match_features(feature_descriptors);

  std::cout << "[Warp cylindrical...]" << std::endl;
  std::cout << "\t" << std::flush;
  for (size_t i = 0; i != image_data.size(); i++) {
    std::cout << "[" << i << "] " << std::flush;
    for (auto& v : match_points[i]) cylindrical_warp_feature_points(v, image_data[i].rows, image_data[i].cols);
    cylindrical_warp_image(image_data[i]);
  }
  std::cout << std::endl;

  auto panoramas_lists = match_images(match_points);

  // draw_matched_features() must before warp_images_together()
  // Since warp_images_together() drops the original image from blending it again
  if (show_matched_features)
    draw_matched_features(image_data, panoramas_lists, match_points);

  warp_images_together(image_data, panoramas_lists);

  return EXIT_SUCCESS;
}
