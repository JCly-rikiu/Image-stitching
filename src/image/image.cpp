#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "image.h"

std::vector<cv::Mat> load_images(std::string& image_dir) {
  std::cout << "[Loading images...]" << std::endl;

  std::vector<cv::Mat> images;

  std::ifstream infile(image_dir + "image_list.txt");
  if (infile.fail()) {
    std::cerr << "Fail to read image_list.txt!" << std::endl;
    return images;
  }

  std::string line;
  while (std::getline(infile, line)) {
    std::istringstream line_stream(line);
    std::string filename;
    if (!(line_stream >> filename)) break;

    cv::Mat image;
    image = cv::imread(image_dir + filename, cv::IMREAD_COLOR);
    if (!image.data) {
      std::cerr << "Could not open or find " << filename << std::endl;
      break;
    }

    images.push_back(image);
  }

  return images;
}
