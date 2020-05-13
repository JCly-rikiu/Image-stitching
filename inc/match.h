#ifndef MATCH_
#define MATCH_

#include <deque>
#include <tuple>
#include <vector>

#include "detection.h"

using MatchPoints = std::vector<std::vector<std::vector<std::tuple<float, float, float, float>>>>;
using PanoramaLists = std::vector<std::deque<std::tuple<int, float, float>>>;

MatchPoints match_features(const std::vector<MSOPDescriptor>&);
PanoramaLists match_images(MatchPoints&);

#endif
