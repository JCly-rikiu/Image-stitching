#ifndef MATCH_
#define MATCH_

#include <deque>
#include <tuple>
#include <vector>

#include "detection.h"

using FeatureMatches = std::vector<std::vector<std::vector<std::tuple<float, float, float, float>>>>;
using PanoramaLists = std::vector<std::deque<std::tuple<int, float, float>>>;

FeatureMatches MatchFeatures(const std::vector<MSOPDescriptors>&);
PanoramaLists MatchImages(FeatureMatches&);

#endif
