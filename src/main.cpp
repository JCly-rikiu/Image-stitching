#include <iostream>
#include <string>

#include "image.h"
#include "detection.h"

int main(int argc, char *argv[]) {
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

  auto feature_points = get_MSOP_features(image_data[0]);

  return EXIT_SUCCESS;
}
