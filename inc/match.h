#ifndef MATCH_
#define MATCH_

#include <vector>

#include "detection.h"

typedef std::vector<std::vector<std::vector<std::tuple<double, double, double, double>>>> MatchPoints;
MatchPoints match_features(const std::vector<MSOPDescriptor>&);

#endif
