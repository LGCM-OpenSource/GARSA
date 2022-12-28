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

#ifndef JACKKNIFE_HPP
#define JACKKNIFE_HPP

#include <vector>
#include <utility>

namespace Jackknife {
  double stddev(const std::vector <double> &x, int n);
  std::pair <double, double> mean_std(const std::vector <double> &x);
  double zscore(const std::vector <double> &x);
  std::pair <double, double> diff_mean_std(std::vector <double> x,
					   const std::vector <double> &x_ref);
  std::pair <double, double> ratioOfSumsMeanStd(const std::vector <double> &x,
						const std::vector <double> &y);
}

#endif
