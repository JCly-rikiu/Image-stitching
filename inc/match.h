#ifndef MATCH_
#define MATCH_

#include <deque>
#include <tuple>
#include <vector>

#include "detection.h"

typedef std::vector<std::vector<std::vector<std::tuple<float, float, float, float>>>> MatchPoints;
typedef std::vector<std::deque<std::tuple<int, float, float>>> PanoramaLists;

MatchPoints match_features(const std::vector<MSOPDescriptor>&);
PanoramaLists match_images(const MatchPoints&);

#endif
