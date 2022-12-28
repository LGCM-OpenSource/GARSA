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
#include <string>
#include <cstdio>

#include "RestrictSnpSet.hpp"
#include "SnpData.hpp"

namespace LMM {

  using std::vector;
  using std::string;

  RestrictSnpSet::RestrictSnpSet(const string &restrictPartnerFile, bool _matchID, int Nautosomes) {
      matchID = _matchID;
      vector <SnpInfo> restrictSnps;
      if (!restrictPartnerFile.empty()) {
	restrictSnps = SnpData::readBimFile(restrictPartnerFile, Nautosomes);
	for (uint64 m = 0; m < restrictSnps.size(); m++)
	  keys.insert(makeKey(restrictSnps[m]));
      }
    }

  string RestrictSnpSet::makeKey(const SnpInfo &snp) const {
    if (matchID) return snp.ID;
    else {
      char buf[20];
      sprintf(buf, "%d,%d", snp.chrom, snp.physpos);
      return string(buf);
    }
  }
  
  bool RestrictSnpSet::isAllowed(const SnpInfo &snp) const {
    return keys.empty() || keys.find(makeKey(snp)) != keys.end();
  }

}
