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

#include <cstdlib>
#include <cmath>

#include "SnpInfo.hpp"

namespace LMM {
  const int SnpInfo::MAX_VC_NUM = 1000;

  bool SnpInfo::isProximal(const SnpInfo &snp2, double genWindow) const {
    return chrom == snp2.chrom && fabs(genpos - snp2.genpos) < genWindow;
  }
  bool SnpInfo::isProximal(const SnpInfo &snp2, int physWindow) const {
    return chrom == snp2.chrom && abs(physpos - snp2.physpos) < physWindow;
  }
}
