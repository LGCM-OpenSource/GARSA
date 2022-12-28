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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>

#include "BoltParEstCV.hpp"
#include "MemoryUtils.hpp"
#include "NumericUtils.hpp"
#include "StatsUtils.hpp"
#include "Jackknife.hpp"

namespace LMM {

  using std::vector;
  using std::string;
  using std::cout;
  using std::cerr;
  using std::endl;

  BoltParEstCV::ParamData::ParamData(double _f2, double _p) : f2(_f2), p(_p) {}

  bool BoltParEstCV::ParamData::operator < (const BoltParEstCV::ParamData &paramData2) const {
    return StatsUtils::zScoreDiff(PVEs, paramData2.PVEs) < -2;
  }

  BoltParEstCV::BoltParEstCV
  (const SnpData& _snpData, const DataMatrix& _covarDataT, const double subMaskIndivs[],
   const vector < std::pair <std::string, DataMatrix::ValueType> > &_covars, int covarMaxLevels,
   bool _covarUseMissingIndic, int mBlockMultX, int Nautosomes)
    : snpData(_snpData), covarDataT(_covarDataT),
      bolt(_snpData, _covarDataT, subMaskIndivs, _covars, covarMaxLevels, _covarUseMissingIndic,
	   mBlockMultX, Nautosomes),
      covars(_covars), covarUseMissingIndic(_covarUseMissingIndic) {
  }

  /**
   * (f2, p) parameter estimation via cross-validation
   * - after each fold, compare PVEs of putative (f2, p) param pairs
   * - eliminate clearly suboptimal param pairs from future folds
   * - stop when only one param pair left
   *
   * return: iterations used in last CV fold
   */
  int BoltParEstCV::estMixtureParams
  (double *f2Est, double *pEst, double *predBoost, const vector <double> &pheno, 
   double logDeltaEst, double sigma2Kest, int CVfoldsSplit, int CVfoldsCompute, bool CVnoEarlyExit,
   double predBoostMin, bool MCMC, int maxIters, double approxLLtol, int mBlockMultX,
   int Nautosomes) const {

    if (CVfoldsCompute <= 0) {
      const int Nwant = 10000, Nrep = bolt.getNused() / CVfoldsSplit + 1;
      CVfoldsCompute = std::min(CVfoldsSplit, (Nwant+Nrep-1) / Nrep);
      cout << "Max CV folds to compute = " << CVfoldsCompute
	   << " (to have > " << Nwant << " samples)" << endl << endl;
    }
    if (CVfoldsCompute > CVfoldsSplit) {
      cerr << "WARNING: CVfoldsCompute > CVfoldsSplit; setting CVfoldsCompute to " << CVfoldsSplit
	   << endl << endl;
      CVfoldsCompute = CVfoldsSplit;
    }

    int usedIters = 0;
      
    // try a fixed set of (f2, p) mixture param pairs
    const int NUM_F2S = 3; const double f2s[NUM_F2S] = {0.5, 0.3, 0.1};
    const int NUM_PS = 6; const double ps[NUM_PS] = {0.5, 0.2, 0.1, 0.05, 0.02, 0.01};
    //const int NUM_PS = 9; const double ps[NUM_PS] = {0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001};

    // all (f2, p) pairs are in play at the start; this list will be pruned after each fold
    // important: first pair corresponds to infinitesimal model
    vector <ParamData> paramDataAll;
    for (int f2i = 0; f2i < NUM_F2S; f2i++)
      for (int pi = 0; pi < NUM_PS; pi++)
	paramDataAll.push_back(ParamData(f2s[f2i], ps[pi]));
      
    const double *maskIndivs = bolt.getMaskIndivs(); // possibly a subset of snpData.maskIndivs
    uint64 Nstride = snpData.getNstride();
    uint64 M = snpData.getM();

    // divide indivs into CVfoldsSplit folds
    vector <int> foldAssignments(Nstride, -1); // -1 for masked indivs
    int indCtr = 0;
    for (uint64 n = 0; n < Nstride; n++)
      if (maskIndivs[n])
	foldAssignments[n] = (indCtr++) % CVfoldsSplit;

    double *foldMaskIndivs = ALIGNED_MALLOC_DOUBLES(Nstride);
    vector <double> baselineMSEs;

    // run OOS pred for each fold in turn
    for (int fold = 0; fold < CVfoldsCompute; fold++) {

      cout << "====> Starting CV fold " << (fold+1) << " <====" << endl << endl;

      // set up fold assignment mask
      for (uint64 n = 0; n < Nstride; n++)
	foldMaskIndivs[n] = (foldAssignments[n] != fold && foldAssignments[n] != -1);
	
      // create Bolt instance for predicting using non-left-out indivs
      int foldCovarMaxLevels = 1<<30; // no need to re-check covar max levels
      const Bolt boltFold(snpData, covarDataT, foldMaskIndivs, covars, foldCovarMaxLevels,
			  covarUseMissingIndic, mBlockMultX, Nautosomes);

      vector <double> PVEs; double baselinePredMSE;
      { // set up arguments and call Bayes-iter
	uint64 B = paramDataAll.size(); // number of remaining (f2, p) pairs in play

	double *phenoResidCovCompVecs = ALIGNED_MALLOC_DOUBLES(B*boltFold.getNCstride());
	boltFold.maskFillCovCompVecs(phenoResidCovCompVecs, &pheno[0], B);

	double *betasTrans = ALIGNED_MALLOC_DOUBLES(M*B);

	uchar *batchMaskSnps = ALIGNED_MALLOC_UCHARS(M*B);
	const uchar *projMaskSnpsFold = boltFold.getProjMaskSnps();
	for (uint64 m = 0; m < M; m++)
	  memset(batchMaskSnps + m*B, projMaskSnpsFold[m], B*sizeof(batchMaskSnps[0]));

	uint64 MprojMaskFold = boltFold.getMprojMask();
	vector <uint64> Ms(B, MprojMaskFold);

	vector <double> logDeltas(B, logDeltaEst);
	vector <double> sigma2Ks(B, sigma2Kest);

	vector <double> varFrac2Ests(B), pEsts(B);
	for (uint64 b = 0; b < B; b++) {
	  varFrac2Ests[b] = paramDataAll[b].f2;
	  pEsts[b] = paramDataAll[b].p;
	}

	// fit the models, one for each (f2, p) pair
	usedIters =
	  boltFold.batchComputeBayesIter(phenoResidCovCompVecs, betasTrans, batchMaskSnps,
					 &Ms[0], &logDeltas[0], &sigma2Ks[0], &varFrac2Ests[0],
					 &pEsts[0], B, MCMC, maxIters, approxLLtol);

	// reset fold assignment mask to prediction indivs
	for (uint64 n = 0; n < Nstride; n++)
	  foldMaskIndivs[n] = (foldAssignments[n] == fold);

	// build predictions and compute PVEs
	PVEs = boltFold.batchComputePredPVEs(&baselinePredMSE, &pheno[0], betasTrans, B,
					     foldMaskIndivs);

	ALIGNED_FREE(batchMaskSnps);
	ALIGNED_FREE(betasTrans);
	ALIGNED_FREE(phenoResidCovCompVecs);
      }

      baselineMSEs.push_back(baselinePredMSE);
      for (uint64 b = 0; b < paramDataAll.size(); b++) {
	paramDataAll[b].PVEs.push_back(PVEs[b]);
	paramDataAll[b].MSEs.push_back(baselinePredMSE * (1-PVEs[b]));
      }

#ifdef VERBOSE
      cout << endl << "Average PVEs obtained by param pairs tested (high to low):" << endl;
      vector < std::pair <double, string> > avgPVEs;
      for (uint64 b = 0; b < paramDataAll.size(); b++) {
	char buf[100]; sprintf(buf, "f2=%g, p=%g", paramDataAll[b].f2, paramDataAll[b].p);
	avgPVEs.push_back(std::make_pair(NumericUtils::mean(paramDataAll[b].PVEs), string(buf)));
	std::sort(avgPVEs.begin(), avgPVEs.end(), std::greater < std::pair <double, string> > ());
      }
      if (avgPVEs.size() <= 5)
	for (uint64 b = 0; b < avgPVEs.size(); b++)
	  printf("%15s: %f\n", avgPVEs[b].second.c_str(), avgPVEs[b].first);
      else {
	for (int b = 0; b < 3; b++)
	  printf("%15s: %f\n", avgPVEs[b].second.c_str(), avgPVEs[b].first);
	printf("%15s\n", "...");
	printf("%15s: %f\n", avgPVEs.back().second.c_str(), avgPVEs.back().first);
      }
      cout << endl;
#endif

      double bestPVE = *std::max_element(PVEs.begin(), PVEs.end());
      uint bestInd = std::max_element(PVEs.begin(), PVEs.end())-PVEs.begin();
      char bestPars[100]; sprintf(bestPars, "f2=%g, p=%g", paramDataAll[bestInd].f2,
				  paramDataAll[bestInd].p);

#ifdef VERBOSE
      cout << "Detailed CV fold results:" << endl;
      printf("  Absolute prediction MSE baseline (covariates only): %g\n", baselinePredMSE);
      if (paramDataAll[0].f2 == 0.5 && paramDataAll[0].p == 0.5)
	printf("  Absolute prediction MSE using standard LMM:         %g\n",
	       paramDataAll[0].MSEs.back());
      printf("  Absolute prediction MSE, fold-best %14s:  %g\n", bestPars,
	     paramDataAll[bestInd].MSEs.back());
      for (uint b = 0; b < paramDataAll.size(); b++) {
	char buf[100]; sprintf(buf, "f2=%g, p=%g", paramDataAll[b].f2, paramDataAll[b].p);
	printf("    Absolute pred MSE using %15s: %f\n", buf, paramDataAll[b].MSEs.back());
      }
      cout << endl;
#endif

      // prune out significantly suboptimal param settings
      if (!CVnoEarlyExit)
	for (int b = paramDataAll.size()-1; b >= 0; b--)
	  for (uint64 b2 = 0; b2 < paramDataAll.size(); b2++)
	    if (paramDataAll[b] < paramDataAll[b2]) {
	      paramDataAll.erase(paramDataAll.begin() + b);
	      break;
	    }
#ifdef VERBOSE
      cout << "====> End CV fold " << (fold+1) << ": " << paramDataAll.size()
	   << " remaining param pair(s) <====" << endl << endl;
#endif

      if (fold == 0) { // set predBoost: 1 - (smallest MSE) / (inf model MSE)
	printf("Estimated proportion of variance explained using inf model: %.3f\n", PVEs[0]);
	*predBoost = 1 - (1-bestPVE) / (1-PVEs[0]);
	printf("Relative improvement in prediction MSE using non-inf model: %.3f\n\n", *predBoost);
	if (*predBoost < predBoostMin && !CVnoEarlyExit) {
	  printf("Exiting CV: non-inf model does not substantially improve prediction\n");
	  break;
	}
      }

      // early exit if only one pair left
      if (paramDataAll.size() == 1) {
	cout << "Finished cross-validation; params sufficiently constrained after "
	     << (fold+1) << " folds" << endl;
	break;
      }
    }

    if (CVnoEarlyExit) {
      cout << "*** Combined results across all folds ***" << endl;
      printf("Baseline MSE:   %g\n", NumericUtils::mean(baselineMSEs));
      vector < std::pair <double, double> > MSEsToBaselineMeanStds(paramDataAll.size());
      for (uint b = 0; b < paramDataAll.size(); b++)
        MSEsToBaselineMeanStds[b] =
	  Jackknife::ratioOfSumsMeanStd(paramDataAll[b].MSEs, baselineMSEs);
      uint bestInd = min_element(MSEsToBaselineMeanStds.begin(), MSEsToBaselineMeanStds.end())
	- MSEsToBaselineMeanStds.begin();
      printf("Pred R^2 and MSE using standard LMM:   %6.3f (%.3f)   %g\n",
	     1-MSEsToBaselineMeanStds[0].first, MSEsToBaselineMeanStds[0].second,
	     NumericUtils::mean(paramDataAll[0].MSEs));
      printf("Pred R^2 and MSE using best non-inf:   %6.3f (%.3f)   %g\n",
	     1-MSEsToBaselineMeanStds[bestInd].first, MSEsToBaselineMeanStds[bestInd].second,
	     NumericUtils::mean(paramDataAll[bestInd].MSEs));
      for (uint b = 0; b < paramDataAll.size(); b++) {
	char buf[100]; sprintf(buf, "f2=%g, p=%g", paramDataAll[b].f2, paramDataAll[b].p);
	printf("  Pred R^2 and MSE using %15s: %6.3f (%.3f)   %g\n", buf,
	       1-MSEsToBaselineMeanStds[b].first, MSEsToBaselineMeanStds[b].second,
	       NumericUtils::mean(paramDataAll[b].MSEs));
      }
      cout << endl;
    }

    // find best PVE; store corresponding f2, p in output params
    double bestMeanPVE = -1e100;
    for (uint64 b = 0; b < paramDataAll.size(); b++) {
      double meanPVE = NumericUtils::mean(paramDataAll[b].PVEs);
      if (meanPVE > bestMeanPVE) {
	bestMeanPVE = meanPVE;
	*f2Est = paramDataAll[b].f2;
	*pEst = paramDataAll[b].p;
      }
    }
      
    cout << "Optimal mixture parameters according to CV: f2 = " << *f2Est
	 << ", p = " << *pEst << endl;

    ALIGNED_FREE(foldMaskIndivs);

    return usedIters;
  }

  // for use in PhenoBuilder to generate random phenotypes
  const Bolt &BoltParEstCV::getBoltRef(void) const {
    return bolt;
  }
}
