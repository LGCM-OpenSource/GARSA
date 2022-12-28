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

#include <cstdio>
#include <cmath>
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>

#include "SnpInfo.hpp"
#include "Jackknife.hpp"
#include "NumericUtils.hpp"
#include "LapackConst.hpp"
#include "Types.hpp"
#include "MemoryUtils.hpp"
#include "LDscoreCalibration.hpp"

namespace LDscoreCalibration {

  using std::vector;
  using std::pair;
  using std::make_pair;
  using std::cout;
  using std::cerr;
  using std::endl;

  double computeMean(const vector <double> &stats, const vector <bool> &maskSnps) {
    double sum = 0; int ctr = 0;
    for (uint64 m = 0; m < stats.size(); m++)
      if (maskSnps[m]) {
	sum += stats[m];
	ctr++;
      }
    return sum / ctr;
  }
  
  double computeLambdaGC(const vector <double> &stats, const vector <bool> &maskSnps) {
    vector <double> goodStats;
    for (uint64 m = 0; m < stats.size(); m++)
      if (maskSnps[m])
	goodStats.push_back(stats[m]);
    sort(goodStats.begin(), goodStats.end());
    return goodStats[goodStats.size()/2] / 0.455;
  }

  vector <bool> removeOutlierWindows
  (const vector <LMM::SnpInfo> &snps, const vector <double> &stats, vector <bool> maskSnps,
   double outlierChisqThresh, bool useGenDistWindow) {
   
    int M = snps.size();
    int numGoodSnpsPreOutlier = 0;
    for (int m = 0; m < M; m++)
      numGoodSnpsPreOutlier += maskSnps[m];
    cout << "# of SNPs passing filters before outlier removal: "
	 << numGoodSnpsPreOutlier << "/" << M << endl;
    int numGoodSnps = numGoodSnpsPreOutlier;
    printf("Masking windows around outlier snps (chisq > %.1f)", outlierChisqThresh);
    cout << endl;

    vector < pair <double, int> > statsSortInd(M);
    for (int m = 0; m < M; m++)
      statsSortInd[m] = make_pair(stats[m], m);
    sort(statsSortInd.begin(), statsSortInd.end());

    for (int s = M-1; s >= 0; s--) { // filter windows around outliers
      if (statsSortInd[s].first > outlierChisqThresh) {
	int m = statsSortInd[s].second;
	for (int m2 = m+1; m2 < M; m2++) {
	  if (useGenDistWindow) { if (!snps[m].isProximal(snps[m2], OUTLIER_GEN_WINDOW)) break; }
	  else { if (!snps[m].isProximal(snps[m2], OUTLIER_PHYS_WINDOW)) break; }
	  if (maskSnps[m2]) { maskSnps[m2] = false; numGoodSnps--; }
	}
	for (int m2 = m-1; m2 >= 0; m2--) {
	  if (useGenDistWindow) { if (!snps[m].isProximal(snps[m2], OUTLIER_GEN_WINDOW)) break; }
	  else { if (!snps[m].isProximal(snps[m2], OUTLIER_PHYS_WINDOW)) break; }
	  if (maskSnps[m2]) { maskSnps[m2] = false; numGoodSnps--; }
	}
	if (numGoodSnps < numGoodSnpsPreOutlier / 2) {
	  cerr << "WARNING: Half of SNPs gone after removing windows near chisq > "
	       << statsSortInd[s].first << endl;
	  cerr << "         Stopping outlier removal" << endl;
	}
      }
      else
	break;
    }
    cout << "# of SNPs remaining after outlier window removal: "
	 << numGoodSnps << "/" << numGoodSnpsPreOutlier << endl;
    
    return maskSnps;
  }

  double computeIntercept
  (const vector <double> &stats, const vector <double> &LDscores,
   const vector <double> &LDscoresChip, const vector <bool> &maskSnps,
   int varianceDegree, double *attenNull1Ptr) {

    double meanStat = computeMean(stats, maskSnps);
    double slopeToCM = (meanStat - 1) / computeMean(LDscores, maskSnps);

    int M = stats.size(), Mfit = 0;
    for (int m = 0; m < M; m++) Mfit += maskSnps[m];
    double *weightedStats = ALIGNED_MALLOC_DOUBLES(Mfit);
    double *weightedRegCols = ALIGNED_MALLOC_DOUBLES(2*Mfit); // sqrt(w) * [all-1s, LDscores]
    
    for (int m = 0, mfit = 0; m < M; m++)
      if (maskSnps[m]) {
	// compute weight
	double weightDblCount = 1 / std::max(1.0, LDscoresChip[m]);
	double weightHeteroskedasticity =
	  varianceDegree == 2 ? NumericUtils::sq(1 / std::max(1.0, 1+slopeToCM*LDscores[m])) :
	  1 / std::max(2.0, 2 + 0.01*LDscores[m]); // Brendan's approximation

	double sqrtWeight = sqrt(weightDblCount * weightHeteroskedasticity);
	weightedStats[mfit] = stats[m] * sqrtWeight;           // LHS: stats
	weightedRegCols[mfit] = sqrtWeight;                    // RHS: all-1s column
	weightedRegCols[Mfit+mfit] = LDscores[m] * sqrtWeight; // RHS: LDscores column
	mfit++;
      }

    // compute least squares fit
    double intercept;
    {
      char _TRANS = 'N';
      int _M = Mfit;
      int _N = 2; // intercept, slope
      int _NRHS = 1; // fitting one regression (one column of B)
      double *_A = &weightedRegCols[0];
      int _LDA = Mfit;
      double *_B = &weightedStats[0];
      int _LDB = Mfit;
      int _LWORK = 4*Mfit; // may not be optimal, but the fitting should be really fast
      double *_WORK = ALIGNED_MALLOC_DOUBLES(_LWORK);
      int _INFO;
      DGELS_MACRO(&_TRANS, &_M, &_N, &_NRHS, _A, &_LDA, _B, &_LDB, _WORK, &_LWORK, &_INFO);
      intercept = _B[0]; //slope = _B[1];
      ALIGNED_FREE(_WORK);
    }
    ALIGNED_FREE(weightedRegCols);
    ALIGNED_FREE(weightedStats);
    if (attenNull1Ptr != NULL) *attenNull1Ptr = (intercept - 1) / (meanStat - 1);
    return intercept;
  }

  vector <double> computeIntercepts
  (const vector <double> &stats, const vector <double> &LDscores,
   const vector <double> &LDscoresChip, const vector <bool> &maskSnps,
   int varianceDegree, int jackBlocks, double *attenNull1Ptr, double *attenNull1StdPtr) {
    
    uint64 M = stats.size();
    vector <uint64> mfit_to_m;
    for (uint64 m = 0; m < M; m++)
      if (maskSnps[m])
	mfit_to_m.push_back(m);
    uint64 Mfit = mfit_to_m.size();
    
    vector <double> interceptJacks(jackBlocks+1), attenNull1Jacks(jackBlocks+1);
    for (uint64 j = 0; j <= (uint64) jackBlocks; j++) {
      vector <bool> jackMaskSnps = maskSnps;
      for (uint64 mfit = 0; mfit < Mfit; mfit++) {
	if (jackBlocks && mfit*jackBlocks / Mfit == j)
	  jackMaskSnps[mfit_to_m[mfit]] = false;
      }
      interceptJacks[j] = computeIntercept(stats, LDscores, LDscoresChip, jackMaskSnps,
					   varianceDegree, &attenNull1Jacks[j]);
    }
    if (attenNull1Ptr != NULL) *attenNull1Ptr = attenNull1Jacks[jackBlocks];
    if (jackBlocks && attenNull1StdPtr != NULL)
      *attenNull1StdPtr = Jackknife::stddev(attenNull1Jacks, jackBlocks);

    return interceptJacks;
  }

  pair <double, double> calibrateStatPair
  (const vector <LMM::SnpInfo> &snps, const vector <double> &statsRef,
   const vector <double> &statsCur, const vector <double> &LDscores,
   const vector <double> &LDscoresChip, double minMAF, int N, double outlierVarFracThresh,
   bool useGenDistWindow, int varianceDegree) {
    
    int M = snps.size();
    vector <bool> maskSnps(M);
    cout << "Filtering to SNPs with chisq stats, LD Scores, and MAF > " << minMAF << endl;
    for (int m = 0; m < M; m++) // perform initial filtering of snps to use in regression
      maskSnps[m] =
	snps[m].MAF >= minMAF &&      // MAF threshold
	statsRef[m] > 0 &&            // ref stat available
	statsCur[m] > 0 &&            // cur stat available
	!std::isnan(LDscores[m]) &&   // LD Score available
	!std::isnan(LDscoresChip[m]); // LD Score weight available
    
    // perform outlier removal
    double outlierChisqThresh = std::max(MIN_OUTLIER_CHISQ_THRESH, N * outlierVarFracThresh);
    vector <bool> noOutlierMaskSnps = removeOutlierWindows(snps, statsRef, maskSnps,
							   outlierChisqThresh, useGenDistWindow);
    
    // perform regressions
    int jackBlocks = SNP_JACKKNIFE_BLOCKS;
    double attenNull1, attenNull1Std;
    vector <double> interceptRefJacks
      = computeIntercepts(statsRef, LDscores, LDscoresChip, noOutlierMaskSnps, varianceDegree,
			  jackBlocks, &attenNull1, &attenNull1Std);
    vector <double> interceptCurJacks
      = computeIntercepts(statsCur, LDscores, LDscoresChip, noOutlierMaskSnps, varianceDegree,
			  jackBlocks, NULL, NULL);
#ifdef VERBOSE
    printf("Intercept of LD Score regression for ref stats:   %.3f (%.3f)\n",
	   interceptRefJacks[jackBlocks], Jackknife::stddev(interceptRefJacks, jackBlocks));
    printf("Estimated attenuation: %.3f (%.3f)\n", attenNull1, attenNull1Std);
    printf("Intercept of LD Score regression for cur stats: %.3f (%.3f)\n",
	   interceptCurJacks[jackBlocks], Jackknife::stddev(interceptCurJacks, jackBlocks));
#endif
    
    vector <double> calibrationFactorJacks(jackBlocks+1);
    for (int j = 0; j <= jackBlocks; j++)
      calibrationFactorJacks[j] = interceptRefJacks[j] / interceptCurJacks[j];
    pair <double, double> calibrationFactorMeanStd = Jackknife::mean_std(calibrationFactorJacks);
    printf("Calibration factor (ref/cur) to multiply by:      %.3f (%.3f)\n",
	   calibrationFactorMeanStd.first, calibrationFactorMeanStd.second);
    fflush(stdout);
    
    return calibrationFactorMeanStd;
  }

  // jackBlocks: 0 for no jackknife
  RegressionInfo calibrateStat
  (const vector <LMM::SnpInfo> &snps, const vector <double> &stats,
   const vector <double> &LDscores, const vector <double> &LDscoresChip, double minMAF, int N,
   double outlierVarFracThresh, bool useGenDistWindow, int varianceDegree, int jackBlocks) {
    
    RegressionInfo info;
    
    int M = snps.size();
    vector <bool> maskSnps(M);
    for (int m = 0; m < M; m++) maskSnps[m] = stats[m] > 0;
    info.mean = computeMean(stats, maskSnps); // compute mean of all good stats with no filtering
    info.lambdaGC = computeLambdaGC(stats, maskSnps);

    cout << "Filtering to SNPs with chisq stats, LD Scores, and MAF > " << minMAF << endl;
    for (int m = 0; m < M; m++) // perform initial filtering of snps to use in regression
      maskSnps[m] =
	snps[m].MAF >= minMAF &&      // MAF threshold
	stats[m] > 0 &&               // stat available
	!std::isnan(LDscores[m]) &&   // LD Score available
	!std::isnan(LDscoresChip[m]); // LD Score weight available
    
    // perform outlier removal
    double outlierChisqThresh = std::max(MIN_OUTLIER_CHISQ_THRESH, N * outlierVarFracThresh);
    vector <bool> noOutlierMaskSnps = removeOutlierWindows(snps, stats, maskSnps,
							   outlierChisqThresh, useGenDistWindow);
    info.noOutlierMean = computeMean(stats, noOutlierMaskSnps);
    
    // perform regression
    vector <double> interceptJacks
      = computeIntercepts(stats, LDscores, LDscoresChip, noOutlierMaskSnps, varianceDegree,
			  jackBlocks, &info.attenNull1, &info.attenNull1Std);
    info.intercept = interceptJacks[jackBlocks];
    if (jackBlocks) info.interceptStd = Jackknife::stddev(interceptJacks, jackBlocks);

    return info;
  }

}
