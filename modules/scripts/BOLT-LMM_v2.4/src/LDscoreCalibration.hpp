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

#ifndef LDSCORECALIBRATION_HPP
#define LDSCORECALIBRATION_HPP

#include <vector>
#include <utility>

#include "SnpInfo.hpp"

namespace LDscoreCalibration {

  const double MIN_OUTLIER_CHISQ_THRESH = 20.0;
  const double OUTLIER_GEN_WINDOW = 0.01; // 1 cM = 0.01 Morgans
  const int OUTLIER_PHYS_WINDOW = 1000000; // 1 Mb
  const int SNP_JACKKNIFE_BLOCKS = 50;

  double computeMean(const std::vector <double> &stats, const std::vector <bool> &maskSnps);
  double computeLambdaGC(const std::vector <double> &stats, const std::vector <bool> &maskSnps);

  std::vector <bool> removeOutlierWindows
  (const std::vector <LMM::SnpInfo> &snps, const std::vector <double> &stats,
   std::vector <bool> maskSnps, double outlierChisqThresh, bool useGenDistWindow);

  double computeIntercept
  (const std::vector <double> &stats, const std::vector <double> &LDscores,
   const std::vector <double> &LDscoresChip, const std::vector <bool> &maskSnps,
   int varianceDegree, double *attenNull1Ptr);

  std::vector <double> computeIntercepts
  (const std::vector <double> &stats, const std::vector <double> &LDscores,
   const std::vector <double> &LDscoresChip, const std::vector <bool> &maskSnps,
   int varianceDegree, int jackBlocks, double *attenNull1Ptr, double *attenNull1StdPtr);

  std::pair <double, double> calibrateStatPair
  (const std::vector <LMM::SnpInfo> &snps, const std::vector <double> &statsRef,
   const std::vector <double> &statsOther, const std::vector <double> &LDscores,
   const std::vector <double> &LDscoresChip, double minMAF, int N,
   double outlierVarFracThresh, bool useGenDistWindow, int varianceDegree);

  struct RegressionInfo {
    double mean;
    double lambdaGC;
    double noOutlierMean;
    double intercept;
    double interceptStd;
    double attenNull1; // attenuation assuming null=1: (int - 1) / (avg fitted chisq - 1)
    double attenNull1Std; // jackknife std
  };

  RegressionInfo calibrateStat
  (const std::vector <LMM::SnpInfo> &snps, const std::vector <double> &stats,
   const std::vector <double> &LDscores, const std::vector <double> &LDscoresChip,
   double minMAF, int N, double outlierVarFracThresh, bool useGenDistWindow, int varianceDegree,
   int jackBlocks);

}

#endif
