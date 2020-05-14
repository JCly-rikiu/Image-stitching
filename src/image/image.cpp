#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "image.h"

std::vector<cv::Mat> LoadImages(std::string& image_dir) {
  std::cout << "[Loading images...]" << std::endl;

  std::vector<std::string> images_filename;
  for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
    std::string filename;
    if (entry.path().extension().compare(".jpg") == 0)
      filename = entry.path().filename();
    else if (entry.path().extension().compare(".JPG") == 0)
      filename = entry.path().filename();
    else
      continue;

    images_filename.push_back(filename);
  }

  std::vector<cv::Mat> images;
#pragma omp parallel for
  for (const auto& filename : images_filename) {
    cv::Mat image = cv::imread(image_dir + filename, cv::IMREAD_COLOR);
    if (!image.data) std::cerr << "Could not open or find " << filename << std::endl;
    images.push_back(image);

#pragma omp critical
    std::cout << "\t[" << images.size() - 1 << "] " << filename << std::endl;
  }

  return images;
}
