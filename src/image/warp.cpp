#include <algorithm>
#include <cmath>
#include <deque>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "match.h"
#include "warp.h"

const float focal_length = 4000.0 / 15.6 * 16;

void CylindricalWarpImage(cv::Mat& image) {
  cv::CylindricalWarper creator;
  auto warper = creator.create(focal_length);

  cv::Mat warp_image;
  cv::Mat K = (cv::Mat_<float>(3, 3) << focal_length, 0, image.cols / 2, 0, focal_length, image.rows / 2, 0, 0, 1);
  auto pos =
      warper->warp(image, K, cv::Mat::eye(cv::Size(3, 3), CV_32F), cv::INTER_LINEAR, cv::BORDER_CONSTANT, warp_image);

  cv::Mat T = (cv::Mat_<float>(2, 3) << 1, 0, pos.x + image.cols / 2, 0, 1, 0);
  cv::warpAffine(warp_image, warp_image, T, image.size());

  image = warp_image;

  /* old code
  cv::Mat warp_image = cv::Mat::zeros(image.size(), image.type());

  int cx = image.cols / 2;
  int cy = image.rows / 2;

  for (int i = 0; i != image.rows; i++) {
    auto row = warp_image.ptr<cv::Vec3b>(i);
    for (int j = 0; j != image.cols; j++) {
      float theta = j - cx;
      float x = std::tan(theta / focal_length) * focal_length;

      float h = i - cy;
      float y = h * std::sqrt(x * x + focal_length * focal_length) / focal_length;

      x += cx;
      int int_x = x;
      float a = x - int_x;

      y += cy;
      int int_y = y;
      float b = y - int_y;

      if (int_x < 0 || int_y < 0 || int_x + 1 >= image.cols || int_y + 1 >= image.rows) continue;

      row[j] = (1 - a) * (1 - b) * image.at<cv::Vec3b>(int_y, int_x) +
               a * (1 - b) * image.at<cv::Vec3b>(int_y, int_x + 1) + a * b * image.at<cv::Vec3b>(int_y + 1, int_x + 1) +
               (1 - a) * b * image.at<cv::Vec3b>(int_y + 1, int_x);
    }
  }

  image = warp_image;
  */
}

void CylindricalWarpFeaturePoints(std::vector<std::tuple<float, float, float, float>>& feature_points, const int rows,
                                  const int cols) {
  auto warp = [&](auto& i, auto& j) {
    int cx = cols / 2;
    int cy = rows / 2;

    float x = j - cx;
    float theta = focal_length * std::atan(x / focal_length);

    float y = i - cy;
    float h = focal_length * y / std::sqrt(x * x + focal_length * focal_length);

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

void AlphaBlend(cv::Mat& panoramas, cv::Mat& temp, const int offset_j, const int left, const int right) {
  for (int j = left; j != right + 1; j++) {
    auto pc = panoramas.col(j);
    auto tc = temp.col(j - offset_j);

    float alpha = static_cast<float>(j - left) / (right + 1 - left);
    pc = (1 - alpha) * pc + alpha * tc;
  }
}

void BlendImage(cv::Mat& panoramas, cv::Mat& temp, const int offset_j, const int current_left, const int current_right,
                const int last_right) {
  const int blend_half_width = 50;

  int remain_left = current_left;
  if (last_right == -1) {
    // First image, no need to blend
    remain_left = current_left;
  } else if (last_right - current_left < blend_half_width * 2) {
    AlphaBlend(panoramas, temp, offset_j, current_left, last_right);
    remain_left = last_right + 1;
  } else {
    int middle_line = (last_right - current_left) / 2 + current_left;
    AlphaBlend(panoramas, temp, offset_j, middle_line - blend_half_width, middle_line + blend_half_width);
    remain_left = middle_line + blend_half_width + 1;
  }

  cv::Rect p_remain_ROI = cv::Rect(remain_left, 0, current_right - remain_left + 1, panoramas.rows);
  cv::Rect t_remain_ROI = cv::Rect(remain_left - offset_j, 0, current_right - remain_left + 1, panoramas.rows);
  temp(t_remain_ROI).copyTo(panoramas(p_remain_ROI));
}

void DriftCorrection(cv::Mat& panoramas, std::deque<std::tuple<int, float, float>>& list,
                     const float drift_correction_value) {
  std::cout << "\tapply drift correction (360 degree paronama detected)" << std::endl;

  cv::Mat shear_mat = (cv::Mat_<float>(2, 3) << 1, 0, 0, drift_correction_value, 1, 0);
  cv::warpAffine(panoramas, panoramas, shear_mat, panoramas.size());

  for (auto& [image, ti, tj] : list) ti += tj * drift_correction_value;
}

cv::Mat CropRectangle(cv::Mat& panoramas, const std::vector<cv::Mat>& image_data,
                      std::deque<std::tuple<int, float, float>>& list) {
  auto [first_image, first_ti, first_tj] = list.front();
  auto [back_image, back_ti, back_tj] = list.back();

  int half_col = image_data[first_image].cols / 2;
  int left = static_cast<int>(std::ceil(first_tj + focal_length * std::atan(-half_col / focal_length) + half_col));
  half_col = image_data[back_image].cols / 2;
  int right = static_cast<int>(std::floor(back_tj + focal_length * std::atan(half_col / focal_length) + half_col));

  int top = 0;
  int bottom = panoramas.rows;
  for (const auto [image, ti, tj] : list) {
    int cx = image_data[image].cols / 2;
    int cy = image_data[image].rows / 2;

    float x = -cx;

    float y = -cy;
    float h = focal_length * y / std::sqrt(x * x + focal_length * focal_length) + cy;
    top = std::max(top, static_cast<int>(std::ceil(h + ti)));

    y = cy;
    h = focal_length * y / std::sqrt(x * x + focal_length * focal_length) + cy;
    bottom = std::min(bottom, static_cast<int>(std::floor(h + ti)));
  }

  return panoramas(cv::Rect(left, top, right + 1 - left, bottom + 1 - top)).clone();
}

void WarpImagesTogether(const std::vector<cv::Mat>& image_data, PanoramaLists& panorama_lists) {
  std::cout << "\n[Blend images...]" << std::endl;

  for (int pano_index = 1; auto& list : panorama_lists) {
    std::cout << "\n[panorama " << pano_index << "]:";

    // Let all ti be positive
    float min_ti = 0;
    for (const auto [image, ti, tj] : list) min_ti = std::min(min_ti, ti);
    float max_ti = 0;
    for (auto& [image, ti, tj] : list) {
      ti -= min_ti;
      max_ti = std::max(max_ti, ti + image_data[image].rows);
    }

    auto [first_image, first_ti, first_tj] = list.front();
    auto [back_image, back_ti, back_tj] = list.back();

    // Drops the original image from blending it again
    bool is_360 = false;
    float drift_correction_value = 1;
    if (first_image == back_image && list.size() > 1) {
      is_360 = true;
      drift_correction_value = (first_ti - back_ti) / (back_tj - first_tj);  // -(back_ti - first_ti)
      list.pop_back();
      std::tie(back_image, back_ti, back_tj) = list.back();
    }

    int rows = static_cast<int>(std::ceil(max_ti));
    int cols = static_cast<int>(std::ceil(back_tj)) + image_data[back_image].cols;
    cv::Mat panoramas = cv::Mat::zeros(cv::Size(cols, rows), CV_8UC3);

    int last_right = -1;
    for (auto [image, ti, tj] : list) {
      std::cout << " -> [" << image << " (" << ti << " " << tj << ")]" << std::flush;

      int half_col = image_data[image].cols / 2;
      float left = focal_length * std::atan(-half_col / focal_length) + half_col;
      float right = focal_length * std::atan(half_col / focal_length) + half_col;

      int offset_j = static_cast<int>(std::floor(tj));
      cv::Mat temp;
      cv::Mat translation_mat = (cv::Mat_<float>(2, 3) << 1, 0, tj - offset_j, 0, 1, ti);
      cv::warpAffine(image_data[image], temp, translation_mat, cv::Size(image_data[image].cols, rows));

      int current_left = static_cast<int>(std::ceil(tj + left));
      int current_right = static_cast<int>(std::floor(tj + right));

      BlendImage(panoramas, temp, offset_j, current_left, current_right, last_right);

      last_right = current_right;
    }
    std::cout << std::endl;

    if (is_360) DriftCorrection(panoramas, list, drift_correction_value);

    std::cout << "\tSave panorama to ./panorama" + std::to_string(pano_index) + ".jpg" << std::endl;
    cv::imwrite("panorama" + std::to_string(pano_index) + ".jpg", panoramas);

    cv::Mat crop = CropRectangle(panoramas, image_data, list);
    std::cout << "\tSave cropped image to ./panorama" + std::to_string(pano_index) + "-crop.jpg" << std::endl;
    cv::imwrite("panorama" + std::to_string(pano_index) + "-crop.jpg", crop);

    pano_index++;
  }
}

void DrawFeatureMatches(const std::vector<cv::Mat>& image_data, const PanoramaLists& panorama_lists,
                        const FeatureMatches& match_points) {
  std::cout << "[Draw matched features...]" << std::endl;

  for (int pano_index = 1; auto& list : panorama_lists) {
    std::cout << "\t[pano " + std::to_string(pano_index) + "]" << std::flush;

    auto list_end = list.end() - 1;  // For OpenMP controlling predicate
#pragma omp parallel for
    for (auto it = list.begin(); it < list_end; it++) {
      auto [image1, ti1, tj1] = *it;
      auto [image2, ti2, tj2] = *(it + 1);

      std::cout << " [" + std::to_string(image1) + "-" + std::to_string(image2) + "]" << std::flush;

      cv::Mat to;
      cv::hconcat(image_data[image1], image_data[image2], to);
      for (const auto [i1, j1, i2, j2] : match_points[image1][image2]) {
        const int x_shift = image_data[image1].cols;

        cv::circle(to, cv::Point2d(j1, i1), 10, cv::Scalar(0, 0, 255), 3);
        cv::circle(to, cv::Point2d(j2 + x_shift, i2), 10, cv::Scalar(0, 0, 255), 3);

        cv::line(to, cv::Point2d(j1, i1), cv::Point2d(j2 + x_shift, i2), cv::Scalar(0, 255, 0), 2);
      }
      cv::imwrite("pano" + std::to_string(pano_index) + "-" + std::to_string(it - list.begin()) + "-" +
                      std::to_string(image1) + "to" + std::to_string(image2) + ".jpg",
                  to);
    }
    std::cout << std::endl;

    pano_index++;
  }
}
