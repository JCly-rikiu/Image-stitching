#include <algorithm>
#include <deque>
#include <iostream>
#include <random>
#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detection.h"
#include "match.h"

MatchPoints match_features(const std::vector<MSOPDescriptor>& feature_descriptors) {
  std::cout << "[Match features...]" << std::endl;

  const float ratio_threshold = 0.65;

  MatchPoints final_match_points(feature_descriptors.size());
  for (auto& v : final_match_points) v.resize(feature_descriptors.size());

  for (size_t image1_i = 0; image1_i != feature_descriptors.size(); image1_i++) {
    for (size_t image2_i = 0; image2_i != image1_i; image2_i++) {
      auto& image1 = feature_descriptors[image1_i];
      auto& image2 = feature_descriptors[image2_i];

      std::vector<std::tuple<float, float, float, float>> match_points1;
      std::vector<std::tuple<float, float, float, float>> match_points2;
      for (size_t layer = 0; layer != image1.size(); layer++) {
        auto& [feature_points1, descriptor1] = image1[layer];
        auto& [feature_points2, descriptor2] = image2[layer];

        cv::flann::GenericIndex<cvflann::L2<float>> index1(descriptor1, cvflann::KDTreeIndexParams());
        cv::flann::GenericIndex<cvflann::L2<float>> index2(descriptor2, cvflann::KDTreeIndexParams());

        cv::Mat_<int> indices2(descriptor1.rows, 2);
        cv::Mat_<float> distances2(descriptor1.rows, 2);
        index2.knnSearch(descriptor1, indices2, distances2, 2, cvflann::SearchParams());

        for (int i = 0; i != descriptor1.rows; i++) {
          auto d2 = distances2.ptr<float>(i);
          if (d2[0] < ratio_threshold * d2[1]) {
            int j = indices2.at<int>(i, 0);
            auto patch2 = descriptor2.row(j);

            std::vector<int> indices1(2);
            std::vector<float> distances1(2);
            index1.knnSearch(patch2, indices1, distances1, 2, cvflann::SearchParams());

            if (distances1[0] < ratio_threshold * distances1[1]) {
              match_points1.emplace_back(std::tuple_cat(feature_points1[i], feature_points2[j]));
              match_points2.emplace_back(std::tuple_cat(feature_points2[j], feature_points1[i]));
            }
          }
        }
      }

      final_match_points[image1_i][image2_i] = match_points1;
      final_match_points[image2_i][image1_i] = match_points2;
    }
  }

  return final_match_points;
}

std::tuple<int, float, float> translation_RANSAC(
    std::vector<std::tuple<float, float, float, float>>& feature_points) {
  const int k_times = 300;
  const int n_sample = 6;
  const float error = 500;

  if (feature_points.size() < n_sample) {
    feature_points.resize(0);
    return {0, 0, 0};
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, feature_points.size() - 1);

  float best_ti = 0, best_tj = 0;
  int max_num_inlier = 0;
  for (int k = 0; k != k_times; k++) {
    float ti = 0, tj = 0;
    for (int n = 0; n != n_sample; n++) {
      auto [i1, j1, i2, j2] = feature_points[dis(gen)];
      ti += i1 - i2;
      tj += j1 - j2;
    }
    ti /= n_sample;
    tj /= n_sample;

    int num_inliers = 0;
    for (const auto [i1, j1, i2, j2] : feature_points) {
      float ei = i2 + ti - i1;
      float ej = j2 + tj - j1;

      if (ei * ei + ej * ej < error) num_inliers++;
    }

    if (num_inliers > max_num_inlier) {
      max_num_inlier = num_inliers;
      best_ti = ti;
      best_tj = tj;
    }
  }

  if (max_num_inlier < 5.9 + 0.22 * feature_points.size()) {
    feature_points.resize(0);
    return {0, best_ti, best_tj};
  }

  float final_ti = 0, final_tj = 0;
  for (auto it = feature_points.begin(); it != feature_points.end();) {
    const auto [i1, j1, i2, j2] = *it;
    float ei = i2 + best_ti - i1;
    float ej = j2 + best_tj - j1;

    if (ei * ei + ej * ej < error) {
      final_ti += i1 - i2;
      final_tj += j1 - j2;
      it++;
    } else {
      it = feature_points.erase(it);
    }
  }
  final_ti /= max_num_inlier;
  final_tj /= max_num_inlier;

  return {max_num_inlier, final_ti, final_tj};
}

void search(const int current_image, const bool left_search,
            const std::vector<std::vector<std::tuple<float, float, int>>>& translations, std::vector<bool>& checked,
            std::deque<std::tuple<int, float, float>>& list) {
  for (const auto [tj, ti, next_image] : translations[current_image]) {
    if (checked[next_image]) continue;
    checked[next_image] = true;
    if (left_search)
      list.emplace_front(next_image, ti, tj);
    else
      list.emplace_back(next_image, ti, tj);
    search(next_image, left_search, translations, checked, list);
  }
}

PanoramaLists match_images(MatchPoints& match_points) {
  std::cout << "[Match images...]" << std::endl;

  auto num_image = match_points.size();

  std::vector<std::vector<std::tuple<float, float, int>>> left_translations(num_image);
  std::vector<std::vector<std::tuple<float, float, int>>> right_translations(num_image);

  for (size_t image1_i = 0; image1_i != num_image; image1_i++) {
    for (size_t image2_i = 0; image2_i != num_image; image2_i++) {
      if (image1_i == image2_i) continue;
      auto [max_i, ti, tj] = translation_RANSAC(match_points[image1_i][image2_i]);

      if (max_i != 0) {
        std::cout << "\timage: " << image1_i << " -> " << image2_i << " : " << max_i << " / "
                  << match_points[image1_i][image2_i].size() << "  ti = " << ti << " tj = " << tj << std::endl;
        if (tj < 0)
          left_translations[image1_i].emplace_back(tj, ti, image2_i);
        else
          right_translations[image1_i].emplace_back(tj, ti, image2_i);
      }
    }
    std::sort(left_translations[image1_i].begin(), left_translations[image1_i].end(),
              std::greater<std::tuple<float, float, int>>());
    std::sort(right_translations[image1_i].begin(), right_translations[image1_i].end());
  }

  std::vector<bool> checked(num_image, false);
  PanoramaLists panorama_lists;
  for (size_t i = 0; i != num_image; i++) {
    if (checked[i]) continue;
    std::deque<std::tuple<int, float, float>> list;
    list.push_back({i, 0, 0});

    // Set checked[current_image_index] later
    // If there exists a 360 degree panoramas, it should generate a double head deque
    // the double head deque activates the drift correction
    // Notice we must search for right translations (connections) first, in order to find the original head
    search(i, false, right_translations, checked, list);
    search(i, true, left_translations, checked, list);
    checked[i] = true;

    // Since we searched for right translations (connections) first, the first head is the original head
    auto zero_it = std::find_if(list.begin(), list.end(), [&](const auto& e) {
      auto [image, ti, tj] = e;
      return image == static_cast<int>(i);
    });

    // Recover the positive translations
    float ri = 0, rj = 0;
    for (auto it = zero_it; it < list.end(); it++) {
      auto& [image, ti, tj] = *it;
      ri += ti;
      rj += tj;
      ti = ri;
      tj = rj;
    }

    // Recover the negative translations
    ri = 0, rj = 0;
    for (auto it = zero_it; it >= list.begin(); it--) {
      auto& [image, ti, tj] = *it;
      ri += ti;
      rj += tj;
      ti = ri;
      tj = rj;
    }

    // Recover all translations
    ri = -ri;
    rj = -rj;
    for (auto& [image, ti, tj] : list) {
      ti += ri;
      tj += rj;
    }

    if (list.size() > 1) panorama_lists.push_back(list);
  }

  std::cout << "\nTotal: " << panorama_lists.size() << " panoramas.\n" << std::endl;
  return panorama_lists;
}
