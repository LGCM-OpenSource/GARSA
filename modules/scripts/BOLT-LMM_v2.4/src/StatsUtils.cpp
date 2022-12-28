/*
   This file is part of the BOLT-LMM linear mixed model software package
   developed by Po-Ru Loh.  Copyright (C) 2014-2022 Harvard University.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <vector>
#include <cmath>
#include <cstdlib>

#include "Types.hpp"
#include "StatsUtils.hpp"
#include "NumericUtils.hpp"

namespace StatsUtils {

  using std::vector;

  double stdDev(const vector <double> &x, bool isPop) {
    uint64 n = x.size();
    if (n <= 1) return NAN;
    double s = 0.0, s2 = 0.0;
    for (uint64 i = 0; i < n; i++) {
      if (std::isnan(x[i])) return NAN;
      if (std::isinf(x[i])) return INFINITY;
      s += x[i];
      s2 += x[i]*x[i];
    }
    if (isPop) return sqrt((s2 - s*s/n) / n);
    else return sqrt((s2 - s*s/n) / (n-1));
  }

  double zScore(const vector <double> &x) {
    return NumericUtils::mean(x) / stdDev(x);
  }

  // zScore for x-y
  double zScoreDiff(vector <double> x, const vector <double> &y) {
    if (x.size() <= 1) return NAN;
    for (uint64 i = 0; i < x.size(); i++)
      x[i] -= y[i];
    return zScore(x);
  }
}
