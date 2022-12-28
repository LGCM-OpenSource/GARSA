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

#ifndef RESTRICTSNPSET_HPP
#define RESTRICTSNPSET_HPP

#include <set>
#include <string>

#include "SnpInfo.hpp"

namespace LMM {

  class RestrictSnpSet {

    bool matchID; // either match by ID or by (chrom, physpos)
    std::set <std::string> keys;

  public:

    RestrictSnpSet(const std::string &restrictPartnerFile, bool _matchID, int Nautosomes);
    std::string makeKey(const SnpInfo &snp) const;
    bool isAllowed(const SnpInfo &snp) const;

  };
}

#endif
