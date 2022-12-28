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

#ifndef SNPINFO_HPP
#define SNPINFO_HPP

#include <string>

namespace LMM {
  class SnpInfo { // note: non-POD b/c of string
  public:
    static const int MAX_VC_NUM;

    int chrom;
    std::string ID;
    double genpos; // Morgans
    int physpos;
    std::string allele1, allele2;
    double MAF; // note: these MAFs are computed on original maskIndivs (pre-QC filtering)
    int vcNum; // 1-based index of variance comp SNP is assigned to (0 = not in GRM; -1 = exclude)

    bool isProximal(const SnpInfo &snp2, double genWindow) const;
    bool isProximal(const SnpInfo &snp2, int physWindow) const;
  };
}

#endif
