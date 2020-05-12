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

  const double ratio_threshold = 0.65;

  MatchPoints final_match_points(feature_descriptors.size());
  for (auto& v : final_match_points) v.resize(feature_descriptors.size());

  for (size_t image1_i = 0; image1_i != feature_descriptors.size(); image1_i++) {
    for (size_t image2_i = 0; image2_i != image1_i; image2_i++) {
      auto& image1 = feature_descriptors[image1_i];
      auto& image2 = feature_descriptors[image2_i];

      std::vector<std::tuple<double, double, double, double>> match_points1;
      std::vector<std::tuple<double, double, double, double>> match_points2;
      for (size_t layer = 0; layer != image1.size(); layer++) {
        auto& [feature_points1, descriptor1] = image1[layer];
        auto& [feature_points2, descriptor2] = image2[layer];

        std::vector<std::tuple<int, int>> first_matches;
        for (size_t i = 0; i != descriptor1.size(); i++) {
          cv::Mat patch_i = descriptor1[i];

          std::vector<std::tuple<int, int>> distances;
          for (size_t j = 0; j != descriptor2.size(); j++) {
            cv::Mat patch_j = descriptor2[j];
            distances.emplace_back(cv::norm(patch_i - patch_j), j);
          }

          std::sort(distances.begin(), distances.end());

          auto [d1, j1] = distances[0];
          auto [d2, j2] = distances[1];
          if (d1 < ratio_threshold * d2) first_matches.emplace_back(i, j1);
        }

        for (auto [f1, f2] : first_matches) {
          cv::Mat patch_j = descriptor2[f2];

          std::vector<std::tuple<int, int>> distances;
          for (size_t i = 0; i != descriptor1.size(); i++) {
            cv::Mat patch_i = descriptor1[i];
            distances.emplace_back(cv::norm(patch_j - patch_i), i);
          }

          std::sort(distances.begin(), distances.end());

          auto [d1, i1] = distances[0];
          auto [d2, i2] = distances[1];
          if (d1 < ratio_threshold * d2 && i1 == f1) {
            // feature points (x, y), not index
            auto [di1, dj1] = feature_points1[f1];
            auto [di2, dj2] = feature_points2[f2];

            match_points1.emplace_back(di1 * std::pow(2, layer), dj1 * std::pow(2, layer), di2 * std::pow(2, layer),
                                       dj2 * std::pow(2, layer));
            match_points2.emplace_back(di2 * std::pow(2, layer), dj2 * std::pow(2, layer), di1 * std::pow(2, layer),
                                       dj1 * std::pow(2, layer));
          }
        }
      }

      final_match_points[image1_i][image2_i] = match_points1;
      final_match_points[image2_i][image1_i] = match_points2;
    }
  }

  return final_match_points;
}

std::tuple<int, double, double> translation_RANSAC(
    const std::vector<std::tuple<double, double, double, double>>& feature_points) {
  const int k_times = 300;
  const int n_sample = 3;
  const double error = 9;

  if (feature_points.size() < n_sample) return {0, 0, 0};

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, feature_points.size() - 1);

  double best_ti = 0, best_tj = 0;
  int max_num_inlier = 0;
  for (int k = 0; k != k_times; k++) {
    double ti = 0, tj = 0;
    for (int n = 0; n != n_sample; n++) {
      auto [i1, j1, i2, j2] = feature_points[dis(gen)];
      ti += i1 - i2;
      tj += j1 - j2;
    }
    ti /= n_sample;
    tj /= n_sample;

    int num_inliers = 0;
    for (auto [i1, j1, i2, j2] : feature_points) {
      double ei = i2 + ti - i1;
      double ej = j2 + tj - j1;

      if (ei * ei + ej * ej < error) num_inliers++;
    }

    if (num_inliers > max_num_inlier) {
      max_num_inlier = num_inliers;
      best_ti = ti;
      best_tj = tj;
    }
  }

  std::cout << "max " << max_num_inlier << " " << feature_points.size() << std::endl;

  if (max_num_inlier < 5.9 + 0.22 * feature_points.size()) return {0, best_ti, best_tj};

  double final_ti = 0, final_tj = 0;
  for (auto [i1, j1, i2, j2] : feature_points) {
    double ei = i2 + best_ti - i1;
    double ej = j2 + best_tj - j1;

    if (ei * ei + ej * ej < error) {
      final_ti += i1 - i2;
      final_tj += j1 - j2;
    }
  }
  final_ti /= max_num_inlier;
  final_tj /= max_num_inlier;

  return {max_num_inlier, final_ti, final_tj};
}

void search(const int current_image, const bool left_search,
            const std::vector<std::vector<std::tuple<double, double, int>>>& translations, std::vector<bool>& checked,
            std::deque<std::tuple<int, double, double>>& list) {
  for (auto [tj, ti, next_image] : translations[current_image]) {
    if (checked[next_image]) continue;

    if (left_search && tj < 0) {
      checked[next_image] = true;
      list.push_front({next_image, ti, tj});
      search(next_image, true, translations, checked, list);
    } else if (!left_search && tj >= 0) {
      checked[next_image] = true;
      list.push_back({next_image, ti, tj});
      search(next_image, false, translations, checked, list);
    }
  }
}

PanoramasLists match_images(const MatchPoints& match_points) {
  std::cout << "[Match images...]" << std::endl;

  auto num_image = match_points.size();

  std::vector<std::vector<std::tuple<double, double, int>>> left_translations(num_image);
  std::vector<std::vector<std::tuple<double, double, int>>> right_translations(num_image);

  for (size_t image1_i = 0; image1_i != num_image; image1_i++) {
    for (size_t image2_i = 0; image2_i != num_image; image2_i++) {
      if (image1_i == image2_i) continue;
      auto [max_i, ti, tj] = translation_RANSAC(match_points[image1_i][image2_i]);

      std::cout << "image: " << image1_i << " " << image2_i << std::endl;
      std::cout << max_i << " " << ti << " " << tj << std::endl;

      if (max_i != 0) {
        if (tj < 0)
          left_translations[image1_i].emplace_back(tj, ti, image2_i);
        else
          right_translations[image1_i].emplace_back(tj, ti, image2_i);
      }
    }
    std::sort(left_translations[image1_i].begin(), left_translations[image1_i].end(),
              std::greater<std::tuple<double, double, int>>());
    std::sort(right_translations[image1_i].begin(), right_translations[image1_i].end());
  }

  std::vector<bool> checked(num_image, false);
  PanoramasLists panoramas_lists;
  for (size_t i = 0; i != num_image; i++) {
    if (checked[i]) continue;
    std::deque<std::tuple<int, double, double>> list;
    list.push_back({i, 0, 0});

    // check the current image index late
    // if there is a 360 degree panoramas, it will generate a double head deque
    // the double head can help the drift corrections
    // notic we must search for right first
    search(i, false, right_translations, checked, list);
    search(i, true, left_translations, checked, list);
    checked[i] = true;

    // since we search for right first, the first head is the original head
    auto zero_it = std::find_if(list.begin(), list.end(), [&](const auto& e) {
      auto [image, ti, tj] = e;
      return image == static_cast<int>(i);
    });

    // recover the positive translations
    double ri = 0, rj = 0;
    for (auto it = zero_it; it < list.end(); it++) {
      auto& [image, ti, tj] = *it;
      ri += ti;
      rj += tj;
      ti = ri;
      tj = rj;
    }

    // recover the negative translations
    ri = 0, rj = 0;
    for (auto it = zero_it; it >= list.begin(); it--) {
      auto& [image, ti, tj] = *it;
      ri += ti;
      rj += tj;
      ti = ri;
      tj = rj;
    }

    // recover all translations
    ri = -ri;
    rj = -rj;
    for (auto& [image, ti, tj] : list) {
      ti += ri;
      tj += rj;
    }

    if (list.size() > 1) panoramas_lists.push_back(list);
  }

  return panoramas_lists;
}
