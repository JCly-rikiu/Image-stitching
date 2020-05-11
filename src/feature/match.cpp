#include <algorithm>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detection.h"
#include "match.h"

MatchPoints match_features(const std::vector<MSOPDescriptor>& feature_descriptors) {
  const double ratio_threshold = 0.65;

  MatchPoints final_match_points(feature_descriptors.size());
  for (auto& v : final_match_points) v.resize(feature_descriptors.size());

  for (size_t image1_i = 0; image1_i != feature_descriptors.size(); image1_i++) {
    for (size_t image2_i = 0; image2_i != image1_i; image2_i++) {
      auto image1 = feature_descriptors[image1_i];
      auto image2 = feature_descriptors[image2_i];

      std::vector<std::tuple<double, double, double, double>> match_points1;
      std::vector<std::tuple<double, double, double, double>> match_points2;
      for (size_t layer = 0; layer != image1.size(); layer++) {
        auto [feature_points1, descriptor1] = image1[layer];
        auto [feature_points2, descriptor2] = image2[layer];

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
