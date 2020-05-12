#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "match.h"
#include "warp.h"

// const double focal_length = 706.0;
const double focal_length = 4000.0 / 15.6 * 16;

void cylindrical_warp_image(cv::Mat& image) {
  cv::Mat warp_image = cv::Mat::zeros(image.size(), image.type());

  int cx = image.cols / 2;
  int cy = image.rows / 2;

  for (int i = 0; i != image.rows; i++) {
    auto row = warp_image.ptr<cv::Vec3b>(i);
    for (int j = 0; j != image.cols; j++) {
      double theta = j - cx;
      double x = std::tan(theta / focal_length) * focal_length;

      double h = i - cy;
      double y = h * std::sqrt(x * x + focal_length * focal_length) / focal_length;

      x += cx;
      int int_x = x;
      double a = x - int_x;

      y += cy;
      int int_y = y;
      double b = y - int_y;

      if (int_x < 0 || int_y < 0 || int_x + 1 >= image.cols || int_y + 1 >= image.rows) continue;

      row[j] = (1 - a) * (1 - b) * image.at<cv::Vec3b>(int_y, int_x) +
               a * (1 - b) * image.at<cv::Vec3b>(int_y, int_x + 1) + a * b * image.at<cv::Vec3b>(int_y + 1, int_x + 1) +
               (1 - a) * b * image.at<cv::Vec3b>(int_y + 1, int_x);
    }
  }

  image = warp_image;
}

void cylindrical_warp_feature_points(std::vector<std::tuple<double, double, double, double>>& feature_points,
                                     const int rows, const int cols) {
  auto warp = [&](auto& i, auto& j) {
    int cx = cols / 2;
    int cy = rows / 2;

    double x = j - cx;
    double theta = focal_length * std::atan(x / focal_length);

    double y = i - cy;
    double h = focal_length * y / std::sqrt(x * x + focal_length * focal_length);

    theta += cx;
    h += cy;

    i = h;
    j = theta;
  };

  for (auto& [i1, j1, i2, j2] : feature_points) {
    warp(i1, j1);
    warp(i2, j2);
  }
}

void alpha_blend(cv::Mat& panoramas, cv::Mat& temp, const int left, const int right, const bool blend) {
  for (int j = left; j != right + 1; j++) {
    auto pc = panoramas.col(j);
    auto tc = temp.col(j);
    for (int i = 0; i != panoramas.rows; i++) {
      auto t = tc.at<cv::Vec3b>(i, 0);
      auto p = pc.at<cv::Vec3b>(i, 0);
      if (p[0] == 0 && p[1] == 0 && p[2] == 0) pc.at<cv::Vec3b>(i, 0) = t;

      if (t[0] != 0 || t[1] != 0 || t[2] != 0) {
        if (blend) {
          double alpha = static_cast<double>(j - left) / (right - left);
          pc.at<cv::Vec3b>(i, 0) = (1 - alpha) * p + alpha * t;
        } else {
          pc.at<cv::Vec3b>(i, 0) = t;
        }
      }
    }
  }
}

void alpha_blend_image(cv::Mat& panoramas, cv::Mat& temp, const int current_left, const int current_right,
                       const int last_right) {
  const int blend_half_width = 150;
  if (last_right == 0) {
    alpha_blend(panoramas, temp, current_left, current_right, false);
  } else if (last_right - current_left < blend_half_width * 2) {
    alpha_blend(panoramas, temp, current_left, last_right, true);
    alpha_blend(panoramas, temp, last_right + 1, current_right, false);
  } else {
    int middle_line = (last_right - current_left) / 2 + current_left;
    alpha_blend(panoramas, temp, middle_line - blend_half_width, middle_line + blend_half_width, true);
    alpha_blend(panoramas, temp, middle_line + blend_half_width + 1, current_right, false);
  }
}

void drift_correction(cv::Mat& panoramas, const std::vector<cv::Mat>& image_data,
                      const std::deque<std::tuple<int, double, double>>& list) {
  std::cout << "\tdrift correction" << std::endl;

  auto [first_image, first_ti, first_tj] = list.front();
  auto [back_image, back_ti, back_tj] = list.back();

  double drift = first_ti - back_ti;
  double length = back_tj - first_tj;

  int half_col = image_data[back_image].cols / 2;
  double left = focal_length * std::atan(-half_col / focal_length) + half_col;
  double right = focal_length * std::atan(half_col / focal_length) + half_col;
  for (int j = static_cast<int>(std::ceil(back_tj + left)); j != static_cast<int>(std::floor(back_tj + right)); j++) {
    auto pc = panoramas.col(j).clone();
    panoramas.col(j).setTo(cv::Scalar(0, 0, 0));
    cv::Mat translation_mat = (cv::Mat_<double>(2, 3) << 1, 0, j, 0, 1, drift);
    cv::warpAffine(pc, panoramas, translation_mat, panoramas.size(), cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
  }
  for (int j = 0; j != static_cast<int>(std::ceil(back_tj + left)); j++) {
    auto pc = panoramas.col(j).clone();
    panoramas.col(j).setTo(cv::Scalar(0, 0, 0));
    cv::Mat translation_mat = (cv::Mat_<double>(2, 3) << 1, 0, j, 0, 1, j / length * drift);
    cv::warpAffine(pc, panoramas, translation_mat, panoramas.size(), cv::INTER_NEAREST, cv::BORDER_TRANSPARENT);
  }
}

cv::Mat crop_rectangle(cv::Mat& panoramas, const std::vector<cv::Mat>& image_data,
                       std::deque<std::tuple<int, double, double>>& list) {
  auto [first_image, first_ti, first_tj] = list.front();
  auto [back_image, back_ti, back_tj] = list.back();

  int half_col = image_data[first_image].cols / 2;
  int left = static_cast<int>(std::ceil(first_tj + focal_length * std::atan(-half_col / focal_length) + half_col));
  half_col = image_data[back_image].cols / 2;
  int right = static_cast<int>(std::floor(back_tj + focal_length * std::atan(half_col / focal_length) + half_col));

  int top = 0;
  int bottom = panoramas.rows;
  for (auto [image, ti, tj] : list) {
    int cx = image_data[image].cols / 2;
    int cy = image_data[image].rows / 2;

    double x = -cx;

    double y = -cy;
    double h = focal_length * y / std::sqrt(x * x + focal_length * focal_length) + cy;
    top = std::max(top, static_cast<int>(std::ceil(h + ti)));

    y = cy;
    h = focal_length * y / std::sqrt(x * x + focal_length * focal_length) + cy;
    bottom = std::min(bottom, static_cast<int>(std::floor(h + ti)));
  }

  return panoramas(cv::Rect(left, top, right + 1 - left, bottom + 1 - top)).clone();
}

void warp_images_together(const std::vector<cv::Mat>& image_data, PanoramasLists& panoramas_lists) {
  std::cout << "\nTotal: " << panoramas_lists.size() << " panoramas.\n" << std::endl;

  for (int pano_index = 1; auto& list : panoramas_lists) {
    // let all ti be positive
    double min_ti = 0;
    for (auto [image, ti, tj] : list) min_ti = std::min(min_ti, ti);
    double max_ti = 0;
    for (auto& [image, ti, tj] : list) {
      ti -= min_ti;
      max_ti = std::max(max_ti, ti + image_data[image].rows);
    }

    std::cout << "[pano " << pano_index << "]:";
    for (auto [image, ti, tj] : list) std::cout << " -> [" << image << " " << ti << " " << tj << "]";
    std::cout << std::endl;

    std::cout << "\tblend images" << std::endl;

    auto [first_image, first_ti, first_tj] = list.front();
    auto [back_image, back_ti, back_tj] = list.back();

    int rows = static_cast<int>(std::ceil(max_ti));
    int cols = static_cast<int>(std::ceil(back_tj)) + image_data[back_image].cols;
    cv::Mat panoramas = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC3);

    int last_right = 0;
    for (auto [image, ti, tj] : list) {
      int half_col = image_data[image].cols / 2;
      double left = focal_length * std::atan(-half_col / focal_length) + half_col;
      double right = focal_length * std::atan(half_col / focal_length) + half_col;

      cv::Mat temp(cv::Size(cols, rows), CV_8UC3);
      cv::Mat translation_mat = (cv::Mat_<double>(2, 3) << 1, 0, tj, 0, 1, ti);
      cv::warpAffine(image_data[image], temp, translation_mat, temp.size(), cv::INTER_LINEAR);

      int current_left = static_cast<int>(std::ceil(tj + left));
      int current_right = static_cast<int>(std::floor(tj + right));

      alpha_blend_image(panoramas, temp, current_left, current_right, last_right);

      last_right = current_right;
    }

    if (first_image == back_image && list.size() > 1) drift_correction(panoramas, image_data, list);

    std::cout << "\tSave panoramas to ./pano" + std::to_string(pano_index) + ".jpg" << std::endl;
    cv::imwrite("pano" + std::to_string(pano_index) + ".jpg", panoramas);

    cv::Mat crop = crop_rectangle(panoramas, image_data, list);
    std::cout << "\tSave crop image to ./pano-crop" + std::to_string(pano_index) + ".jpg" << std::endl;
    cv::imwrite("pano-crop" + std::to_string(pano_index) + ".jpg", crop);

    pano_index++;
  }
}
