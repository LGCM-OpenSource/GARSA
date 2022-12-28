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

#include <cmath>
#include <cstring>
#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>
#include <set>
#include <algorithm>
#include <numeric>
#include <utility>

#include "omp.h"
#include "zlib.h"

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/math/distributions/chi_squared.hpp>

#include "Types.hpp"
#include "Timer.hpp"
#include "SnpData.hpp"
#include "CovariateBasis.hpp"
#include "NumericUtils.hpp"
#include "LapackConst.hpp"
#include "MemoryUtils.hpp"
#include "Jackknife.hpp"
#include "LDscoreCalibration.hpp"
#include "MapInterpolater.hpp"
#include "FileUtils.hpp"
#include "StatsUtils.hpp"
#include "Bolt.hpp"

namespace LMM {

  using std::vector;
  using std::string;
  using std::pair;
  using std::cout;
  using std::cerr;
  using std::endl;
  using FileUtils::getline;

  const double Bolt::BAD_SNP_STAT = -1e9;

  Bolt::StatsDataRetroLOCO::StatsDataRetroLOCO
  (const std::string &_statName, const std::vector <double> &_stats,
   const std::vector < std::vector <double> > &_calibratedResids,
   const std::vector < std::pair <uint64, int> > &_snpChunkEnds,
   const std::vector <double> &_VinvScaleFactors=vector <double>()) :
    statName(_statName), stats(_stats), calibratedResids(_calibratedResids),
    snpChunkEnds(_snpChunkEnds), VinvScaleFactors(_VinvScaleFactors) {}

  /**
   * for marker m:
   * - snpData.buildMaskedSnpVector(with lut0129=0129)
   * - compute mean of non-missing
   * - replace (non-masked) missing values with mean; subtract mean?; compute norm2
   * - compute components onto covBasis.basis (dot prod or dgemv)
   * - normalize mean-centered SNP (before projecting out other covars) to mean sq entry 1:
   *   - meanCenterNorm2 = norm2-(comp0sq)        <-- normalize to Nused
   *   - projNorm2       = norm2-SUM(components2)
   * ==> save snpValueLookup[m] := 0, 1/meanCenterNorm, 2/meanCenterNorm, mean/meanCenterNorm
   * ==> save Xnorm2s[m] := Nused * projNorm/meanCenterNorm
   *                     (= Nused if only all-1s covar, < Nused o/w)
   *
   * returns 1 if snp is good, 0 if snp is bad:
   * - no non-missing
   * - nothing left after projecting out covars
   *   (happens e.g. if non-missing are monomorphic on indivs to use: all-1s will be projected out)
   */
  uchar Bolt::initMarker(uint64 m, double snpVector[]) {
    // compute mean of non-missing (in masking set)
    double sumGenoNonMissing = 0; int numGenoNonMissing = 0;
    for (uint64 n = 0; n < Nstride; n++) {
      if (maskIndivs[n] // important! don't use masked-out values
	  && snpVector[n] != 9) {
	sumGenoNonMissing += snpVector[n];
	numGenoNonMissing++;
      }
    }
    
    // snp is bad: no non-missing
    if (numGenoNonMissing == 0) return 0;

    // mean-center and replace missing values with mean (centered to 0)
    double mean = sumGenoNonMissing / numGenoNonMissing;
    for (uint64 n = 0; n < Nstride; n++) {
      if (maskIndivs[n]) {
	if (snpVector[n] == 9)
	  snpVector[n] = 0;
	else
	  snpVector[n] -= mean;
      }
      else
	assert(snpVector[n] == 0); // buildMaskedSnpVector should've already zeroed
    }

    // compute components onto covBasis.basis (negate later when scaling by projNorm)
    // no need to zero out components after Cindep: snpCovBasisNegComps already 0-initialized
    covBasis.computeCindepComponents(snpCovBasisNegComps + m*Cstride, snpVector);

    // normalize mean-centered SNP (before projecting out other covars) to mean sq entry 1

    double meanCenterNorm2 = NumericUtils::norm2(snpVector, Nstride);
    // normalize to Nused-1 (dimensionality of subspace with all-1s vector projected out)
    double invMeanCenterNorm = sqrt((Nused-1) / meanCenterNorm2);

    // calculate projNorm^2 = norm^2 - SUM(components^2)
    double projNorm2 = meanCenterNorm2;
    for (uint64 c = 0; c < Cindep; c++) // subtract out components along other covariates
      projNorm2 -= NumericUtils::sq(snpCovBasisNegComps[m*Cstride + c]);

    // snp is bad: nothing left after projecting covars
    if (projNorm2 < 0.1) return 0; // check against 0.1 to take into account rounding error!

    // save lookup of 0129 values: 0, 1/meanCenterNorm, 2/meanCenterNorm, mean/meanCenterNorm
    snpValueLookup[m][0] = -mean * invMeanCenterNorm;
    snpValueLookup[m][1] = (1-mean) * invMeanCenterNorm;
    snpValueLookup[m][2] = (2-mean) * invMeanCenterNorm;
    snpValueLookup[m][3] = 0;

    // negate and scale down basis components (orig computed with 0129 values) by meanCenterNorm
    for (uint64 c = 0; c < Cindep; c++) {
      snpCovBasisNegComps[m*Cstride + c] *= -invMeanCenterNorm;
    }
    
    // save square norm of column of X (i.e., normalized SNP)
    Xnorm2s[m] = projNorm2 * NumericUtils::sq(invMeanCenterNorm);

    return 1;
  }

  void Bolt::maskFillCovCompVecs(double covCompVecs[], const double vec[], uint64 B) const {
    // create first vec: zero-out ignored indivs
    for (uint64 n = 0; n < Nstride; n++)
      covCompVecs[n] = maskIndivs[n] * vec[n];

    // compute components along cov basis vectors and put in [Nstride..Nstride+Cstride)
    covBasis.computeCindepComponents(covCompVecs + Nstride, vec);
    // important! zero out values for non-existent components
    for (uint64 c = Cindep; c < Cstride; c++)
      covCompVecs[Nstride+c] = 0;

    // copy to fill rest of B x (Nstride+Cstride) matrix
    for (uint64 b = 1; b < B; b++)
      memcpy(covCompVecs + b*(Nstride+Cstride), covCompVecs,
	     (Nstride+Cstride)*sizeof(covCompVecs[0]));
  }

  // covComps: B x Cstride
  void Bolt::fillCovComps(double covComps[], const double vec[], uint64 B) const {    
    double *covCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    maskFillCovCompVecs(covCompVec, vec, 1); // mask vec in work array; compute components
    for (uint64 b = 0; b < B; b++) // copy over components
      memcpy(covComps + b*Cstride, covCompVec + Nstride, Cstride*sizeof(covComps[0]));
    ALIGNED_FREE(covCompVec);
  }

  double Bolt::dotCovCompVecs(const double xCovCompVec[], const double yCovCompVec[]) const {
    return NumericUtils::dot(xCovCompVec, yCovCompVec, Nstride) -
      NumericUtils::dot(xCovCompVec + Nstride, yCovCompVec + Nstride, Cstride);
  }

  double Bolt::computeProjNorm2(const double xCovCompVec[]) const {
    return NumericUtils::norm2(xCovCompVec, Nstride) -
      NumericUtils::norm2(xCovCompVec + Nstride, Cstride); // take off cov comps
  }

  void Bolt::computeProjNorm2s(double projNorm2s[], const double xCovCompVecs[], uint64 B) const {
    for (uint64 b = 0; b < B; b++)
      projNorm2s[b] = computeProjNorm2(xCovCompVecs + b*(Nstride+Cstride));
  }

  vector <int> Bolt::makeChunkAssignments(int numLeaveOutChunks) const {
    cout << "Assigning SNPs to " << numLeaveOutChunks << " chunks for leave-out analysis" << endl;
    cout << "Each chunk is excluded when testing SNPs belonging to the chunk" << endl;

    vector <int> chunkAssignments(M, -1);

    if (numLeaveOutChunks <= numChromsProjMask) { // distribute chroms to make chunks even
      const vector <SnpInfo> &snps = snpData.getSnpInfo();
      std::map <int, uint64> chromPops;
      for (uint64 m = 0; m < M; m++)
	if (projMaskSnps[m])
	  chromPops[snps[m].chrom]++;
      uint64 batchPops[numLeaveOutChunks];
      memset(batchPops, 0, numLeaveOutChunks*sizeof(batchPops[0]));
      std::map <int, int> chromBatch;
      vector < std::pair <uint64, int> > popChroms;
      for (std::map <int, uint64>::iterator it = chromPops.begin(); it != chromPops.end(); it++)
	popChroms.push_back(std::make_pair(it->second, it->first));
      std::sort(popChroms.begin(), popChroms.end(), std::greater < std::pair <uint64, int> > ());
      for (uint64 i = 0; i < popChroms.size(); i++) {
	int b = std::min_element(batchPops, batchPops+numLeaveOutChunks) - batchPops;
	chromBatch[popChroms[i].second] = b;
	batchPops[b] += popChroms[i].first;
      }
      /*
      cout << "chrom batch assignments:";
      for (std::map <int, int>::iterator it = chromBatch.begin(); it != chromBatch.end(); it++)
	cout << " " << it->second;
      cout << endl;
      cout << "batch pops:" << endl;
      for (int b = 0; b < numLeaveOutChunks; b++)
	cout << b << ": " << batchPops[b] << endl;
      */
      for (uint64 m = 0; m < M; m++)
	if (projMaskSnps[m])
	  chunkAssignments[m] = chromBatch[snps[m].chrom];
    }
    else { // divide genome up equally
      for (uint64 m = 0, mGood = 0; m < M; m++) // mGood runs up to MprojMask
	if (projMaskSnps[m]) { // evenly divide the ***non-bad*** snps!
	  uint64 b = (uint64) numLeaveOutChunks * mGood / MprojMask;
	  chunkAssignments[m] = b;
	  mGood++;
	}
    }
    /*
    vector < std::pair <uint64, int> > snpChunkEnds = computeSnpChunkEnds(chunkAssignments);
    int testChr[5] = {10, 17, 18, 19, 20}, testBp[5] = {1234, 89087, 123498170, 13417983, 1497};
    for (int i = 0; i < 5; i++)
      cout << testChr[i] << " " << testBp[i] << ": "
	   << findChunkAssignment(snpChunkEnds, testChr[i], testBp[i]) << endl;
    */
    return chunkAssignments;
  }

  vector < std::pair <uint64, int> >
  Bolt::computeSnpChunkEnds(const vector <int> &chunkAssignments) const {
    vector < std::pair <uint64, int> > snpChunkEnds;
    uint64 mPrev = 0; int chunkPrev = -1;
    for (uint64 m = 0; m < M; m++)
      if (chunkAssignments[m] >= 0) {
	if (chunkAssignments[m] != chunkPrev) { // transition
	  if (chunkPrev != -1)
	    snpChunkEnds.push_back(std::make_pair(mPrev, chunkPrev));
	  snpChunkEnds.push_back(std::make_pair(m, chunkAssignments[m]));
	}
	mPrev = m; chunkPrev = chunkAssignments[m];
      }
    snpChunkEnds.push_back(std::make_pair(mPrev, chunkPrev));
    unique(snpChunkEnds.begin(), snpChunkEnds.end());
    /*
    const vector <SnpInfo> &snps = snpData.getSnpInfo();
    for (uint i = 0; i < snpChunkEnds.size(); i++)
      printf("chr %d, bp %d: chunk %d\n", snps[snpChunkEnds[i].first].chrom,
	     snps[snpChunkEnds[i].first].physpos, snpChunkEnds[i].second);
    */
    return snpChunkEnds;
  }

  int Bolt::findChunkAssignment(const vector < std::pair <uint64, int> > &snpChunkEnds,
				int chr, int bp) const {
    const vector <SnpInfo> &snps = snpData.getSnpInfo();
    uint64 closestDist = 1LL<<60; int chunk = -1;
    for (uint i = 0; i < snpChunkEnds.size(); i++) {
      uint64 m = snpChunkEnds[i].first;
      uint64 dist = ((uint64) (abs(snps[m].chrom - chr))<<30) + abs(snps[m].physpos - bp);
      if (dist < closestDist) {
	closestDist = dist;
	chunk = snpChunkEnds[i].second;
      }
    }
    return chunk;
  }

  /**
   * fills batchMaskSnps[] with leave-out masking, leaving out windows around masked chunks
   *
   * batchMaskSnps[]: (out) M x B
   * chunkAssignments[]: (in) M-vector of leave-out chunk assignments; -1 for masked-out snps
   * chunks[]: (in) B-vector of chunks corresponding to columns of batchMaskSnps
   *
   * return: Mused = sum(batchMaskSnps)
   */
  vector <uint64> Bolt::makeBatchMaskSnps(uchar batchMaskSnps[],
					  const vector <int> &chunkAssignments,
					  const vector <int> &chunks, double genWindow,
					  int physWindow) const {
    uint64 B = chunks.size();

    // find first and last snp in each batch
    vector <uint64> mFirst(B, M), mLast(B);
    for (uint64 m = 0; m < M; m++)
      if (projMaskSnps[m]) {
	int c = chunkAssignments[m];
	for (uint64 b = 0; b < B; b++)
	  if (chunks[b] == c) {
	    if (mFirst[b] == M) mFirst[b] = m;
	    mLast[b] = m;
	  }
      }

    vector <uint64> Mused(B);
    for (uint64 m = 0; m < M; m++)
      for (uint64 b = 0; b < B; b++) {
	bool in_batch = projMaskSnps[m] && (chunkAssignments[m] != chunks[b])
	  && !snpData.isProximal(m, mFirst[b], genWindow, physWindow)
	  && !snpData.isProximal(m, mLast[b], genWindow, physWindow);
	batchMaskSnps[m*B + b] = in_batch;
	Mused[b] += in_batch;
      }

    //for (uint64 b = 0; b < B; b++)
    //  cout << "Mused[" << b << "]: " << Mused[b] << " " << Mused[b] / (double) MprojMask << endl;
    return Mused;
  }

  vector <uint64> Bolt::selectProSnps(int numCalibSnps, const double HinvPhiCovCompVec[], int seed)
    const {
    const double chisqMax = 5; // only choose snps with GRAMMAR stat < chisqMax
    vector <uint64> mProSnps(numCalibSnps, M);

    // divide good snps up into numCalibSnps blocks
    vector <uint64> mFirst(numCalibSnps+1, M); // starts of blocks
    for (uint64 m = 0, mGood = 0; m < M; m++) // mGood runs up to MprojMask
      if (projMaskSnps[m]) { // evenly divide the ***non-bad*** snps!
	uint64 j = (uint64) numCalibSnps * mGood / MprojMask;
	if (mFirst[j] == M) mFirst[j] = m;
	mGood++;
      }

    boost::mt19937 rng(seed+321);
    boost::uniform_int<> unif(0, 1<<30);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > randu(rng, unif);

    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(256 * sizeof(*work));
    double *xNegCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    double HinvPhiNorm2 = computeProjNorm2(HinvPhiCovCompVec);

    int numTried = 0;
    for (int j = 0; j < numCalibSnps; j++) {
      while (mProSnps[j] == M) {
	uint64 m = mFirst[j] + randu() % (mFirst[j+1]-mFirst[j]);
	if (projMaskSnps[m]) { // make sure we don't choose a bad snp!
	  numTried++;
	  
	  buildMaskedSnpNegCovCompVec(xNegCovCompVec, m, work);
	  double retroStat =
	    NumericUtils::sq(NumericUtils::dot(HinvPhiCovCompVec, xNegCovCompVec, Nstride+Cstride))
	    / HinvPhiNorm2 / Xnorm2s[m] * (Nused-Cindep);

	  if (retroStat < chisqMax) // require GRAMMAR stat not outlier
	    mProSnps[j] = m;
	}
      }
    }
    
    cout << "Selected " << numCalibSnps << " SNPs for computation of prospective stat" << endl;
    cout << "Tried " << numTried << "; threw out " << numTried-numCalibSnps
	 << " with GRAMMAR chisq > " << chisqMax << endl;

    ALIGNED_FREE(xNegCovCompVec);
    ALIGNED_FREE(work);

    return mProSnps;
  }

  /**
   * multiply v * X and transpose; i.e., multiply X' * v' (subtracting covComp dot products)
   * explicitly: X_proj' * v_proj', where [.]_proj = [.] - [.]covComps*covBasis
   *
   * double X[]: N x M (implicitly stored in Bolt as M x Nstride+Cstride)
   * double vCovCompVecs[]: (in) B x Nstride+Cstride
   * double XtransVecs[]: (out) M x B = X'(+negCovCompVecs) * vCovCompVecs'
   * - for LAPACK column-major, need to compute:
   *   XtransVecs' =     vCovCompVecs    *   X'(+negCovCompVecs)'
   *      B x M      B x Nstride+Cstride     Nstride+Cstride x M
   *                     TRANSA = 'T'            TRANSB = 'N'
   *
   * todo: 32/64-byte strides (extend B to Bstride divisible by 4/8)?
   */
  void Bolt::multXtrans(double XtransVecs[], const double vCovCompVecs[], uint64 B)
    const {
    
    double *snpCovCompVecBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * (Nstride+Cstride));
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(omp_get_max_threads() * 256*sizeof(*work));
    for (uint64 m0 = 0; m0 < M; m0 += mBlockMultX) {
      uint64 mBlockMultXCrop = std::min(M, m0+mBlockMultX) - m0;
#pragma omp parallel for
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (projMaskSnps[m]) // build snp vector + sign-flipped covar comps
	  buildMaskedSnpNegCovCompVec(snpCovCompVecBlock + mPlus * (Nstride+Cstride), m,
				      work + (omp_get_thread_num()<<8));
	else
	  memset(snpCovCompVecBlock + mPlus * (Nstride+Cstride), 0,
		 (Nstride+Cstride) * sizeof(snpCovCompVecBlock[0]));
      }
#ifdef MEASURE_DGEMM
      //Timer timer;
      unsigned long long tsc = Timer::rdtsc();
#endif
      {
	char TRANSA_ = 'T';
	char TRANSB_ = 'N';
	int M_ = B;
	int N_ = mBlockMultXCrop;
	int K_ = Nstride+Cstride;
	double ALPHA_ = 1.0;
	const double *A_ = vCovCompVecs;
	int LDA_ = Nstride+Cstride;
	double *B_ = snpCovCompVecBlock;
	int LDB_ = Nstride+Cstride;
	double BETA_ = 0.0;
	double *C_ = XtransVecs + m0*B; // TODO: need to be aligned?
	int LDC_ = B;

	DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		    &BETA_, C_, &LDC_);
      }
#ifdef MEASURE_DGEMM
      dgemmTicks += Timer::rdtsc() - tsc;
      //dgemmTicks += timer.update_time();
#endif
    }
    ALIGNED_FREE(work);
    ALIGNED_FREE(snpCovCompVecBlock);
  }

  /**
   * multiply X * Xtvt and transpose; i.e., multiply Xtvt' * X' (including positive covComps)
   * in practice, we compute (X * (X'*v'))' = (v*X) * X'
   *
   * double X[]: N x M (implicitly stored in Bolt as M x Nstride+Cstride)
   * double XtransVecs[]: (in) M x B
   * double vCovCompVecs[]: (out) B x Nstride+Cstride = XtransVecs' * X'(+CovCompVecs)
   * - for LAPACK column-major, need to compute:
   *      vCovCompVecs'    =  X'(+CovCompVecs)'   *   XtransVecs
   *   Nstride+Cstride x B   Nstride+Cstride x M        M x B
   *                            TRANSA = 'N'         TRANSB = 'T'
   */
  void Bolt::multX(double vCovCompVecs[], const double XtransVecs[], uint64 B)
    const {

    memset(vCovCompVecs, 0, B * (Nstride+Cstride) * sizeof(vCovCompVecs[0]));

    double *snpCovCompVecBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * (Nstride+Cstride));
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(omp_get_max_threads() * 256*sizeof(*work));
    for (uint64 m0 = 0; m0 < M; m0 += mBlockMultX) {
      uint64 mBlockMultXCrop = std::min(M, m0+mBlockMultX) - m0;
#pragma omp parallel for
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (projMaskSnps[m]) // build snp vector + covar comps
	  buildMaskedSnpCovCompVec(snpCovCompVecBlock + mPlus * (Nstride+Cstride), m,
				   work + (omp_get_thread_num()<<8));
	else
	  memset(snpCovCompVecBlock + mPlus * (Nstride+Cstride), 0,
		 (Nstride+Cstride) * sizeof(snpCovCompVecBlock[0]));
      }
#ifdef MEASURE_DGEMM
      //Timer timer;
      unsigned long long tsc = Timer::rdtsc();
#endif
      {
	char TRANSA_ = 'N';
	char TRANSB_ = 'T';
	int M_ = Nstride+Cstride;
	int N_ = B;
	int K_ = mBlockMultXCrop;
	double ALPHA_ = 1.0;
	double *A_ = snpCovCompVecBlock;
	int LDA_ = Nstride+Cstride;
	const double *B_ = XtransVecs + m0*B; // TODO: need to be aligned?
	int LDB_ = B;
	double BETA_ = 1.0;
	double *C_ = vCovCompVecs;
	int LDC_ = Nstride+Cstride;
	DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		    &BETA_, C_, &LDC_);
      }
#ifdef MEASURE_DGEMM
      dgemmTicks += Timer::rdtsc() - tsc;
      //dgemmTicks += timer.update_time();
#endif
    }
    ALIGNED_FREE(work);
    ALIGNED_FREE(snpCovCompVecBlock);
  }  
  
  /**
   * multiply v * XX' (with positive covComps in input and output)
   * explicitly: X_proj X_proj' * v_proj', where [.]_proj = [.] - [.]covComps*covBasis
   *
   * double vCovCompVecs[]: (in/out) B x Nstride+Cstride
   */
  void Bolt::multXXtransMask(double vCovCompVecs[], const uchar batchMaskSnps[], uint64 B) const {
    double *XtransVecs = ALIGNED_MALLOC_DOUBLES(M * B);
    
    //Timer timer;

    // multiply v * X (or equivalently, X' * v')
    multXtrans(XtransVecs, vCovCompVecs, B);
    //cout << "Time for multXtrans = " << timer.update_time() << endl;

    // apply M x B mask
    for (uint64 mb = 0; mb < M*B; mb++)
      XtransVecs[mb] *= batchMaskSnps[mb];
    //cout << "Time for mask = " << timer.update_time() << endl;
    
    // multiply X * (masked X' * v'), or equivalently, (masked (v * X)) * X'
    multX(vCovCompVecs, XtransVecs, B);
    //cout << "Time for multX = " << timer.update_time() << endl;
    
    ALIGNED_FREE(XtransVecs);
  }

  /**
   * H_proj * x_proj, where H_proj = X_proj X_proj' / M + delta * I_proj
   * note that I * x_proj = I_proj * x_proj for x_proj orthogonal to covBasis
   */
  void Bolt::multH(double HmultCovCompVecs[], const double xCovCompVecs[],
		   const uchar batchMaskSnps[], const double logDeltas[], const uint64 Ms[],
		   uint64 B) const {
    memcpy(HmultCovCompVecs, xCovCompVecs, B * (Nstride+Cstride) * sizeof(HmultCovCompVecs[0]));
    multXXtransMask(HmultCovCompVecs, batchMaskSnps, B); // Hmult <- X[b]*X[b]'*x[b] (temp)
    for (uint64 bnc = 0, b = 0; b < B; b++) {
      double invM = 1 / (double) Ms[b], delta = exp(logDeltas[b]);
      for (uint64 nc = 0; nc < Nstride+Cstride; nc++, bnc++)
	HmultCovCompVecs[bnc] = invM * HmultCovCompVecs[bnc] + delta * xCovCompVecs[bnc];
    }
  }
  
  /**
   * solves a batch of B equations
   *     H_proj[b] * x_proj[b] = b_proj[b]
   * where
   *     H_proj[b] = X_proj[b]*X_proj[b]'/Ms[b] + deltas[b] * I_proj
   * and X_proj[b] denotes X with some columns masked out according to batchMaskSnps[:,b]
   * (projections of x, b, and columns of X are implicitly represented via covComps)
   *
   * double xCovCompVecs[]: (in/out) B x Nstride+Cstride
   * double bCovCompVecs[]: (in) B x Nstride+Cstride
   * double batchMaskSnps[]: (in) M x B
   */
  void Bolt::conjGradSolve(double xCovCompVecs[], bool useStartVecs, const double bCovCompVecs[],
			   const uchar batchMaskSnps[], const uint64 Ms[],
			   const double logDeltas[], uint64 B, int maxIters, double CGtol) const {
#ifdef VERBOSE
    Timer timer;
    cout << "  Batch-solving " << B << " systems of equations using conjugate gradient iteration"
	 << endl;
#endif
#ifdef MEASURE_DGEMM
    unsigned long long tscStart = Timer::rdtsc();
    dgemmTicks = 0;
#endif

    const uint64 BxNC = B * (Nstride+Cstride);

    vector <double> r2orig(B), r2olds(B), r2news(B);
    computeProjNorm2s(&r2orig[0], bCovCompVecs, B);

    double *rCovCompVecs = ALIGNED_MALLOC_DOUBLES(BxNC);
    double *HmultCovCompVecs = ALIGNED_MALLOC_DOUBLES(BxNC);
    if (useStartVecs) {
      multH(HmultCovCompVecs, xCovCompVecs, batchMaskSnps, logDeltas, Ms, B); // H*x
      for (uint64 bnc = 0; bnc < BxNC; bnc++)
	rCovCompVecs[bnc] = bCovCompVecs[bnc] - HmultCovCompVecs[bnc]; // r=b-H*x
      computeProjNorm2s(&r2olds[0], rCovCompVecs, B); // rsold=r'*r
    }
    else { // starting at x=0
      memset(xCovCompVecs, 0, BxNC * sizeof(rCovCompVecs[0]));
      memcpy(rCovCompVecs, bCovCompVecs, BxNC * sizeof(rCovCompVecs[0]));
      r2olds = r2orig;
    }

    double *pCovCompVecs = ALIGNED_MALLOC_DOUBLES(BxNC);
    memcpy(pCovCompVecs, rCovCompVecs, BxNC * sizeof(pCovCompVecs[0])); // p=r

    for (int iter = 0; iter < maxIters; iter++) {
      //covBasis.printProj(pCovCompVecs + (B-1)*(Nstride+Cstride), "pCovCompVecs end");
      multH(HmultCovCompVecs, pCovCompVecs, batchMaskSnps, logDeltas, Ms, B); // H*p

      for (uint64 bnc = 0, b = 0; b < B; b++) {
	double *p = pCovCompVecs + b * (Nstride+Cstride);
	double *Hp = HmultCovCompVecs + b * (Nstride+Cstride);
	
	// alpha=rsold/(p'*Ap)
	double alpha = r2olds[b] / dotCovCompVecs(p, Hp);

	for (uint64 nc = 0; nc < Nstride+Cstride; nc++, bnc++) {
	  xCovCompVecs[bnc] += alpha * pCovCompVecs[bnc]; //x=x+alpha*p
	  rCovCompVecs[bnc] -= alpha * HmultCovCompVecs[bnc]; //r=r-alpha*Ap
	}
      }
      
      computeProjNorm2s(&r2news[0], rCovCompVecs, B); // rsnew=r'*r
      
#ifdef VERBOSE
      double min_rRatio = 1e9, max_rRatio = 0;
      for (uint64 b = 0; b < B; b++) {
	double rRatio = sqrt(r2news[b] / r2orig[b]);
	min_rRatio = std::min(rRatio, min_rRatio);
	max_rRatio = std::max(rRatio, max_rRatio);
      }

      vector <double> resNorm2s(B);
      computeProjNorm2s(&resNorm2s[0], xCovCompVecs, B);
      for (uint64 b = 0; b < B; b++)
	resNorm2s[b] *= NumericUtils::sq(exp(logDeltas[b]));

      printf("  iter %d:  time=%.2f  rNorms/orig: (%.1g,%.1g)  res2s: %g..%g\n",
	     iter+1, timer.update_time(), min_rRatio, max_rRatio, resNorm2s[0], resNorm2s[B-1]);
      fflush(stdout);
#endif

      // check convergence
      bool converged = true;
      for (uint64 b = 0; b < B; b++)
	if (sqrt(r2news[b] / r2orig[b]) > CGtol)
	  converged = false;
      if (converged) {
	cout << "  Converged at iter " << iter+1 << ": rNorms/orig all < CGtol=" << CGtol << endl;
	//for (uint64 b = 0; b < B; b++)
	//  covBasis.printProj(xCovCompVecs + b*(Nstride+Cstride), "xCovCompVecs");
	break;
      }

      for (uint64 bnc = 0, b = 0; b < B; b++) {
	double r2ratio = r2news[b] / r2olds[b];
	for (uint64 nc = 0; nc < Nstride+Cstride; nc++, bnc++)
	  pCovCompVecs[bnc] = rCovCompVecs[bnc] + r2ratio * pCovCompVecs[bnc]; // p=r+rsnew/rsold*p
      }

      r2olds = r2news; // rsold=rsnew
    }
    
    ALIGNED_FREE(pCovCompVecs);
    ALIGNED_FREE(HmultCovCompVecs);
    ALIGNED_FREE(rCovCompVecs);
#ifdef MEASURE_DGEMM
    double dgemmPct = 100 * dgemmTicks / (double) (Timer::rdtsc()-tscStart);
    printf("  Time breakdown: dgemm = %.1f%%, memory/overhead = %.1f%%\n", dgemmPct, 100-dgemmPct);
    fflush(stdout);
#endif
  }

  void Bolt::applySwaps(double x[], const vector < std::pair <uint64, uint64> > &swaps) const {
    for (uint64 s = 0; s < swaps.size(); s++)
      std::swap(x[swaps[s].first], x[swaps[s].second]);
  }
  void Bolt::undoSwaps(double x[], const vector < std::pair <uint64, uint64> > &swaps) const {
    for (int s = (int) swaps.size()-1; s >= 0; s--)
      std::swap(x[swaps[s].first], x[swaps[s].second]);
  }
  void Bolt::swapCovCompVecs(double covCompVec1[], double covCompVec2[], double tmp[]) const {
    memcpy(tmp, covCompVec1, (Nstride+Cstride)*sizeof(tmp[0]));
    memcpy(covCompVec1, covCompVec2, (Nstride+Cstride)*sizeof(tmp[0]));
    memcpy(covCompVec2, tmp, (Nstride+Cstride)*sizeof(tmp[0]));
  }

  /**
   * double yResidCovCompVecs: (in/out) B x (Nstride+Cstride) (zero-filled for ignored indivs)
   *   last Cstride elements of each row contain components of phi_resid along covbasis vecs
   * double betasTrans[]: (in/out) M x B
   * uchar batchMaskSnps[]: (in) M x B -- subsets of projMaskSnps
   * double logDeltas[], sigma2Ks[]: (in) B -- VCs
   *
   * return: iterations used
  */
  int Bolt::batchComputeBayesIter(double yResidCovCompVecs[], double betasTrans[],
				  const uchar batchMaskSnps[], const uint64 Ms[],
				  const double logDeltas[], const double sigma2Ks[],
				  double varFrac2Ests[], double pEsts[], uint64 B, bool MCMC,
				  int maxIters, double approxLLtol) const {
#ifdef VERBOSE
    Timer timer;
    cout << "  Beginning " << (MCMC ? "Gibbs sampling" : "variational Bayes") << endl;
#endif

#ifdef MEASURE_DGEMM
    unsigned long long tscStart = Timer::rdtsc();
    dgemmTicks = 0;
#endif

    int usedIters = maxIters;
    
    for (uint64 b = 0; b < B; b++)
      if (pEsts[b] == 0 || pEsts[b] == 1) { // infinitesimal model; avoid division by 0
	pEsts[b] = 0.5;
	varFrac2Ests[b] = 0.5;
      }

    // initialize betas
    memset(betasTrans, 0, M*B*sizeof(betasTrans[0]));
    
    double *betaBarsTrans = NULL;
    if (MCMC) { // initialize aggregate betas for MCMC
      betaBarsTrans = ALIGNED_MALLOC_DOUBLES(M*B);
      memset(betaBarsTrans, 0, M*B * sizeof(betaBarsTrans[0]));
    }

    boost::mt19937 rng(0); // rng for MCMC... TODO: remove if not doing MCMC
    boost::normal_distribution<> nd(0.0, 1.0);
    boost::variate_generator<boost::mt19937&, 
			     boost::normal_distribution<> > randn(rng, nd);
    boost::variate_generator<boost::mt19937&,
			     boost::uniform_01<> > rand(rng, boost::uniform_01<>());

    double sigma2es[B], sigma2beta1s[B], sigma2beta2s[B];
    for (uint64 b = 0; b < B; b++) {
      sigma2es[b] = exp(logDeltas[b]) * sigma2Ks[b];
      sigma2beta1s[b] = sigma2Ks[b]/Ms[b] * (1-varFrac2Ests[b]) / pEsts[b];
      sigma2beta2s[b] = sigma2Ks[b]/Ms[b] * varFrac2Ests[b] / (1-pEsts[b]);
    }

    vector <bool> converged(B); uint64 Bleft = B; // number of un-converged computations in batch
    // un-converged yResids will be swapped forward in the batch to reduce work from B to Bleft
    vector < std::pair <uint64, uint64> > swaps; // save sequence of swaps
    vector <uint64> origInds(B);
    for (uint64 b = 0; b < B; b++)
      origInds[b] = b;

    vector <double> approxLLs(B), approxLLsPrev;
    //double beta_m_updates[B];
    double resNorm2s[B];
    double *snpCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride); // TODO: eventually remove
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(omp_get_max_threads() * 256*sizeof(*work));

    // some good SNPs will be left out from all batches if LD pruning is performed
    // TODO: implement corresponding speedup in CG functions (multX, multH, etc.) for standard LMM?
    uchar *anyBatchMaskSnps = ALIGNED_MALLOC_UCHARS(M);
    memset(anyBatchMaskSnps, 0, M * sizeof(anyBatchMaskSnps[0]));
    for (uint64 m = 0; m < M; m++)
      for (uint64 b = 0; b < B; b++)
	if (batchMaskSnps[m*B + b])
	  anyBatchMaskSnps[m] = 1;

    // for block updates
    double *snpNegCovCompVecBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * (Nstride+Cstride));
    double *snpDots = ALIGNED_MALLOC_DOUBLES(M * mBlockMultX); // [m][mPlus] = dot prod(m, m+mPlus)
    double *XtransResids = ALIGNED_MALLOC_DOUBLES(mBlockMultX * B);
    double *betaBlockUpdates = ALIGNED_MALLOC_DOUBLES(mBlockMultX * B);

    int usedItersMCMC = 0;
    bool allConverged;

    for (int iter = 0; iter < maxIters; iter++) {

      bool useIterMCMC = true; // TODO: discard first 20% of iters as burn-in?

      if (!MCMC) {
	for (uint64 b = 0; b < B; b++) // constant terms in approx LLs
	  if (!converged[b])
	    approxLLs[b] = -((double) Nused - Cindep)/2 * log(2*M_PI*sigma2es[b]);
      }
      else
	if (useIterMCMC) usedItersMCMC++;
      
      /***** BEGIN BIG ITERATION *****/
      
      for (uint64 m0 = 0; m0 < M; m0 += mBlockMultX) { // similar to multXtrans()

	uint64 mBlockMultXCrop = std::min(M, m0+mBlockMultX) - m0;
#pragma omp parallel for
	for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	  uint64 m = m0+mPlus;
	  if (projMaskSnps[m] && anyBatchMaskSnps[m]) // build snp vec + sign-flipped covar comps
	    buildMaskedSnpNegCovCompVec(snpNegCovCompVecBlock + mPlus * (Nstride+Cstride), m,
					work + (omp_get_thread_num()<<8));
	  else
	    memset(snpNegCovCompVecBlock + mPlus * (Nstride+Cstride), 0,
		   (Nstride+Cstride) * sizeof(snpNegCovCompVecBlock[0]));
	}
#ifdef MEASURE_DGEMM
	//Timer timer;
	unsigned long long tsc = Timer::rdtsc();
#endif
	{
	  char TRANSA_ = 'T';
	  char TRANSB_ = 'N';
	  int M_ = Bleft;
	  int N_ = mBlockMultXCrop;
	  int K_ = Nstride+Cstride;
	  double ALPHA_ = 1.0;
	  double *A_ = yResidCovCompVecs;
	  int LDA_ = Nstride+Cstride;
	  double *B_ = snpNegCovCompVecBlock;
	  int LDB_ = Nstride+Cstride;
	  double BETA_ = 0.0;
	  double *C_ = XtransResids; // note: need to mult by 1/Xnorm2s[m] to get beta_m_updates
	  int LDC_ = B;

	  DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		      &BETA_, C_, &LDC_);
	}
	if (iter == 0) { // calculate dot products of snps in block 
	  char TRANSA_ = 'T';
	  char TRANSB_ = 'N';
	  int M_ = mBlockMultXCrop;
	  int N_ = mBlockMultXCrop;
	  int K_ = Nstride; // no cov comps!
	  double ALPHA_ = 1.0;
	  double *A_ = snpNegCovCompVecBlock;
	  int LDA_ = Nstride+Cstride;
	  double *B_ = snpNegCovCompVecBlock;
	  int LDB_ = Nstride+Cstride;
	  double BETA_ = 0.0;
	  double *C_ = snpDots + m0*mBlockMultX; // not crop!  fill rows [m0..m0+mBlockMultXCrop)
	  int LDC_ = mBlockMultX; // not crop!

	  DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		      &BETA_, C_, &LDC_);

	  K_ = Cstride; // now subtract out the cov comp dot products!
	  ALPHA_ = -1.0;
	  A_ += Nstride;
	  B_ += Nstride;
	  BETA_ = 1.0;

	  DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		      &BETA_, C_, &LDC_);
	}
#ifdef MEASURE_DGEMM
	dgemmTicks += Timer::rdtsc() - tsc;
	//dgemmTicks += timer.update_time();
#endif

	/***** BEGIN LITTLE ITER *****/

	for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {

	  uint64 m = m0+mPlus;

	  double *beta_m_updates = betaBlockUpdates + mPlus*B;

	  if (!(projMaskSnps[m] && anyBatchMaskSnps[m])) {
	    memset(beta_m_updates, 0, B * sizeof(betaBlockUpdates[0]));
	    continue;
	  }

#ifdef SINGLE_M_UPDATE
	  // TODO: just a check; eventually remove
	  // compute dot prods of snpCovCompVec with permuted yResidCovCompVecs (DGEMV)
	  buildMaskedSnpNegCovCompVec(snpCovCompVec, m, work);
	  {	
	    char TRANS_ = 'T';
	    int M_ = Nstride+Cstride;
	    int N_ = Bleft; // only run on un-converged batch elements (permuted to first Bleft)
	    double ALPHA_ = 1/Xnorm2s[m];
	    double *A_ = yResidCovCompVecs;
	    int LDA_ = Nstride+Cstride;
	    double *X_ = snpCovCompVec;
	    int INCX_ = 1;
	    double BETA_ = 0;
	    double *Y_ = beta_m_updates; // permuted; need to un-swap
	    int INCY_ = 1;
	    DGEMV_MACRO(&TRANS_, &M_, &N_, &ALPHA_, A_, &LDA_, X_, &INCX_, &BETA_, Y_, &INCY_);
	  }
	  for (uint64 b = 0; b < Bleft; b++) {
	    if (fabs(beta_m_updates[b] - XtransResids[mPlus*B + b]/Xnorm2s[m]) > 1e-9) {
	      cerr << "ERROR: " << m << endl;
	      cout << beta_m_updates[b] - XtransResids[mPlus*B + b]/Xnorm2s[m] << " ";
	      exit(1);
	    }
	  }
#else
	  for (uint64 b = 0; b < Bleft; b++)
	    beta_m_updates[b] = XtransResids[mPlus*B + b] / Xnorm2s[m];
#endif

	  // undo swaps to beta_m_updates
	  undoSwaps(beta_m_updates, swaps);

	  double *betas_m = betasTrans + m*B;

	  // update to the residual is beta_old -
	  //   ([dot prod / Xnorm2s[m], currently in beta_m_updates] + beta_old) * BLUP_shrink
	  for (uint64 b = 0; b < B; b++) {
	    if (converged[b])          // not actually necessary with permutation, but makes
	      beta_m_updates[b] = 0.0; // behavior consistent when turning off permutation
	    else {
	      /*
	      double beta_new = batchMaskSnps[m*B + b] ? // mask left-out SNPs
	      (beta_m_updates[b] + betas_m[b]) * blupShrinks[b] : 0.0;
	      */
	      double beta_new = 0.0;
	      if (batchMaskSnps[m*B + b]) {
		double beta_hat_m = beta_m_updates[b] + betas_m[b];

		// probably not worth precomputing and storing these in M x B matrices
		double s1 = sqrt(sigma2beta1s[b] + sigma2es[b]/Xnorm2s[m]);
		double s2 = sqrt(sigma2beta2s[b] + sigma2es[b]/Xnorm2s[m]);
		double shrink1 = sigma2beta1s[b] / NumericUtils::sq(s1);
		double shrink2 = sigma2beta2s[b] / NumericUtils::sq(s2);

		double exponent1 = -0.5*NumericUtils::sq(beta_hat_m/s1);
		double exponent2 = -0.5*NumericUtils::sq(beta_hat_m/s2);
		double exponentMax = std::max(exponent1, exponent2);
		exponent1 -= exponentMax; // TODO: one of these will be 0; can optimize
		exponent2 -= exponentMax;
		double p1 = pEsts[b]/s1 * exp(exponent1);
		double p2 = (1-pEsts[b])/s2 * exp(exponent2);
		double p_m = p1 / (p1+p2);

		double mu1 = beta_hat_m * shrink1;
		double mu2 = beta_hat_m * shrink2;

		double beta_mean = p_m * mu1 + (1-p_m) * mu2;

		double tau1sq = 1 / (1/sigma2beta1s[b] + 1/(sigma2es[b]/Xnorm2s[m]));
		double tau2sq = 1 / (1/sigma2beta2s[b] + 1/(sigma2es[b]/Xnorm2s[m]));

		if (!MCMC)
		  beta_new = beta_mean;
		else {
		  // if past burn-in, sample the marginal posterior mean (Rao-Blackwell)
		  if (useIterMCMC) betaBarsTrans[m*B + b] += beta_mean;
		  // sample from the posterior for the actual MCMC update
		  if (rand() < p_m)
		    beta_new = mu1 + randn() * sqrt(tau1sq);
		  else
		    beta_new = mu2 + randn() * sqrt(tau2sq);
		}

		if (!MCMC) {
		  // compute approx LL contribution (equivalently, penalty for beta_new)
		  double mu1sq = NumericUtils::sq(mu1);
		  double mu2sq = NumericUtils::sq(mu2);
		  double var_q = p_m * (tau1sq + mu1sq) + (1-p_m) * (tau2sq + mu2sq)
		    - NumericUtils::sq(beta_new);

		  double KL = 0;
		  if (p_m != 0) KL += p_m * log(p_m / pEsts[b]);
		  if (1-p_m != 0) KL += (1-p_m) * log((1-p_m) / (1-pEsts[b]));
		  KL -= p_m/2 * (1+log(tau1sq/sigma2beta1s[b]) - (tau1sq+mu1sq)/sigma2beta1s[b])
		    + (1-p_m)/2 * (1+log(tau2sq/sigma2beta2s[b]) - (tau2sq+mu2sq)/sigma2beta2s[b]);
	      
		  double penalty = Xnorm2s[m] / (2*sigma2es[b]) * var_q + KL;
		  approxLLs[b] -= penalty;
		}
	      }

	      beta_m_updates[b] = betas_m[b] - beta_new;
	      betas_m[b] = beta_new;
	    }
	  }

	  // re-apply swaps to beta_m_updates
	  applySwaps(beta_m_updates, swaps);

#ifdef SINGLE_M_UPDATE
	  // sign-flip covar comps in snpCovCompVec
	  for (uint64 n = Nstride; n < Nstride+Cstride; n++)
	    snpCovCompVec[n] *= -1;

	  // update yResidCovCompVecs: add beta updates * snpCovCompVec (DGER)
	  {
	    int M_ = Nstride+Cstride;
	    int N_ = Bleft;
	    double ALPHA_ = 1;
	    double *X_ = snpCovCompVec;
	    int INCX_ = 1;
	    double *Y_ = beta_m_updates;
	    int INCY_ = 1;
	    double *A_ = yResidCovCompVecs;
	    int LDA_ = Nstride+Cstride;
	    DGER_MACRO(&M_, &N_, &ALPHA_, X_, &INCX_, Y_, &INCY_, A_, &LDA_);
	  }
#endif
	  // update XtransResids according to how yResidCovCompVecs would have been updated
	  for (uint64 mPlus2 = mPlus+1; mPlus2 < mBlockMultXCrop; mPlus2++) {
	    double dot12 = snpDots[(m0+mPlus)*mBlockMultX + mPlus2];
	    for (uint64 b = 0; b < Bleft; b++)
	      XtransResids[mPlus2*B + b] += beta_m_updates[b] * dot12;
	  }
	}

	/***** END LITTLE ITER *****/

#ifndef SINGLE_M_UPDATE
	// perform block update of yResidCovCompVecs using coeffs in betaBlockUpdates
	// similar to multX()
	{
#ifdef MEASURE_DGEMM
	  //Timer timer;
	  tsc = Timer::rdtsc();
#endif
	  char TRANSA_ = 'N';
	  char TRANSB_ = 'T';
	  int M_ = Nstride; // no cov comps!
	  int N_ = Bleft;
	  int K_ = mBlockMultXCrop;
	  double ALPHA_ = 1.0;
	  double *A_ = snpNegCovCompVecBlock;
	  int LDA_ = Nstride+Cstride;
	  double *B_ = betaBlockUpdates;
	  int LDB_ = B; // only first Bleft cols used
	  double BETA_ = 1.0;
	  double *C_ = yResidCovCompVecs;
	  int LDC_ = Nstride+Cstride;
	  DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		      &BETA_, C_, &LDC_);

	  M_ = Cstride; // now subtract out the negated cov comps!
	  ALPHA_ = -1.0;
	  A_ += Nstride;
	  C_ += Nstride;
	  DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		      &BETA_, C_, &LDC_);

#ifdef MEASURE_DGEMM
	  dgemmTicks += Timer::rdtsc() - tsc;
	  //dgemmTicks += timer.update_time();
#endif
	}
#endif
      }
      /***** END BIG ITERATION *****/

      computeProjNorm2s(resNorm2s, yResidCovCompVecs, Bleft); // for approx LLs
      undoSwaps(resNorm2s, swaps); // resNorm2s are initially permuted; need to undo swaps

      double minApproxLLdiff = 1e9, maxApproxLLdiff = -1e9;
      if (!MCMC) {
	for (uint64 b = 0; b < B; b++)
	  if (!converged[b]) {
	    approxLLs[b] -= resNorm2s[b] / (2*sigma2es[b]);
	    if (iter > 0) {
	      minApproxLLdiff = std::min(minApproxLLdiff, approxLLs[b] - approxLLsPrev[b]);
	      maxApproxLLdiff = std::max(maxApproxLLdiff, approxLLs[b] - approxLLsPrev[b]);
	    }
	  }
	/*
	if (iter > 0) {
	  cout << "approxLL diffs at iter " << iter+1 << ":";
	  for (uint64 b = 0; b < B; b++) {
	    if (!converged[b]) printf(" %.2f", approxLLs[b] - approxLLsPrev[b]);
	    else printf("  -- ");
	  }
	  cout << endl;
	}
	*/
      }

#ifdef VERBOSE
      cout << "  iter " << iter+1; if (MCMC) cout << " of " << maxIters;
      printf(":  time=%.2f for %2d active reps", timer.update_time(), (int) Bleft);
      if (!MCMC && iter > 0)
	printf("  approxLL diffs: (%.2f,%.2f)", minApproxLLdiff, maxApproxLLdiff);
      cout << endl;
#endif

      if (!MCMC && iter > 0) { // check convergence
	allConverged = true;
	for (uint64 b = 0; b < B; b++) 
	  if (!converged[b]) {
	    if (approxLLs[b] - approxLLsPrev[b] < approxLLtol) {
	      converged[b] = true;
	      Bleft--;
	    }
	    else
	      allConverged = false;
	  }	
	if (allConverged) {
	  cout << "  Converged at iter " << iter+1 << ": approxLL diffs each have been < LLtol="
	       << approxLLtol << endl;
	  usedIters = iter+1;
	  break;
	}
	// check if any of 0..Bleft is now converged; if so, swap yResids and record the swap
	for (uint64 b1 = 0; b1 < B; b1++)
	  if (converged[origInds[b1]])
	    for (uint64 b2 = B-1; b2 > b1; b2--)
	      if (!converged[origInds[b2]]) {
		//cout << "swapping positions (" << b1 << "," << b2 << "); original indices "
		//     << origInds[b1] << " and " << origInds[b2] << endl;
		std::swap(origInds[b1], origInds[b2]);
		swapCovCompVecs(yResidCovCompVecs + b1*(Nstride+Cstride),
				yResidCovCompVecs + b2*(Nstride+Cstride), snpCovCompVec);
		swaps.push_back(std::make_pair(b1, b2));
		break;
	      }
      }
      approxLLsPrev = approxLLs;
    }

    if (!MCMC) { // undo swaps of yResids
      if (!allConverged) {
	cerr << "WARNING: Iteration limit reached before approxLLtol; VB may not have converged\n"
	     << "         Increasing --maxIters may improve phenotype model and statistical power"
	     << endl;
      }
      for (int s = (int) swaps.size()-1; s >= 0; s--)
	swapCovCompVecs(yResidCovCompVecs + swaps[s].first*(Nstride+Cstride),
			yResidCovCompVecs + swaps[s].second*(Nstride+Cstride), snpCovCompVec);
    }
    else {
#ifdef VERBOSE
      cout << "Finished " << maxIters << " MCMC iters; averaging over " << usedItersMCMC << endl;
#endif
      // normalize betaBars from sums to averages of aggregated betas
      for (uint64 mb = 0; mb < M*B; mb++)
	betaBarsTrans[mb] /= usedItersMCMC;
      // update yResids with differences between current betas and betaBars... todo: DGEMM
      double *beta_m_updates = ALIGNED_MALLOC_DOUBLES(B);
      for (uint64 m = 0; m < M; m++)
	if (projMaskSnps[m] && anyBatchMaskSnps[m]) {
	  buildMaskedSnpCovCompVec(snpCovCompVec, m, work);
	  for (uint64 b = 0; b < B; b++)
	    beta_m_updates[b] = betasTrans[m*B + b] - betaBarsTrans[m*B + b];
	  // update yResidCovCompVecs: add beta updates * snpCovCompVec (DGER)
	  {
	    int M_ = Nstride+Cstride;
	    int N_ = B; // Bleft in similar block above, but no swapping is ever done
	    double ALPHA_ = 1;
	    double *X_ = snpCovCompVec;
	    int INCX_ = 1;
	    double *Y_ = beta_m_updates;
	    int INCY_ = 1;
	    double *A_ = yResidCovCompVecs;
	    int LDA_ = Nstride+Cstride;
	    DGER_MACRO(&M_, &N_, &ALPHA_, X_, &INCX_, Y_, &INCY_, A_, &LDA_);
	  }
	}
      ALIGNED_FREE(beta_m_updates);
      // overwrite betas with betaBars for output
      memcpy(betasTrans, betaBarsTrans, M*B*sizeof(betasTrans[0]));
      ALIGNED_FREE(betaBarsTrans);
    }

#ifdef MEASURE_DGEMM
    double dgemmPct = 100 * dgemmTicks / (double) (Timer::rdtsc()-tscStart);
    printf("  Time breakdown: dgemm = %.1f%%, memory/overhead = %.1f%%\n", dgemmPct, 100-dgemmPct);
    fflush(stdout);
#endif

    ALIGNED_FREE(betaBlockUpdates);
    ALIGNED_FREE(XtransResids);
    ALIGNED_FREE(anyBatchMaskSnps);
    ALIGNED_FREE(snpDots);
    ALIGNED_FREE(snpNegCovCompVecBlock);
    ALIGNED_FREE(work);
    ALIGNED_FREE(snpCovCompVec);

    return usedIters;
  }

  Bolt::Bolt(const SnpData &_snpData, const DataMatrix &_covarDataT, const double _maskIndivs[],
	     const vector < pair <string, DataMatrix::ValueType> > &covars, int covMaxLevels,
	     bool covarUseMissingIndic, int _mBlockMultX, int _Nautosomes) :
    snpData(_snpData), covarDataT(_covarDataT),
    covBasis(_covarDataT, _maskIndivs, covars, covMaxLevels, covarUseMissingIndic),
    mBlockMultX(_mBlockMultX), Nautosomes(_Nautosomes) { // mBlockMultX = block size for X, X' mult in CG... TODO: optimize for speed
    init();
  }

  /**
   * creates lookup tables of 0129 translation as well as covbasis components for each SNP
   * creates sub-mask to eliminate any bad snps
   */
  void Bolt::init(void) {
    cout << endl << "=== Initializing Bolt object: projecting and normalizing SNPs ===" << endl
	 << endl;

    M = snpData.getM();
    Nstride = snpData.getNstride();
    maskIndivs = covBasis.getMaskIndivs();
    Nused = covBasis.getNused();
    Cindep = covBasis.getCindep();
    //Cstride = (Nstride+Cindep+7)&~7 - Nstride; // 64-byte alignment of covCompVecs
    Cstride = (Cindep+3)&~3;

    // error-check that maskIndivs from covBasis is subset of maskIndivs from snpData
    for (uint64 n = 0; n < Nstride; n++)
      if (maskIndivs[n] && !snpData.getMaskIndivs()[n]) {
	cerr << "ERROR (internal): maskIndivs from covBasis must be subset of maskIndivs "
	  "from snpData" << endl;
	exit(1);
      }

    Xnorm2s = ALIGNED_MALLOC_DOUBLES(M);
    snpValueLookup = (double (*)[4]) ALIGNED_MALLOC(M * sizeof(*snpValueLookup));
    // zero-fill; prevent arithmetic exceptions for ignored m's
    memset(Xnorm2s, 0, M*sizeof(Xnorm2s[0]));
    memset(snpValueLookup, 0, M*sizeof(snpValueLookup[0]));

    snpCovBasisNegComps = ALIGNED_MALLOC_DOUBLES(M*Cstride);
    memset(snpCovBasisNegComps, 0, M*Cstride*sizeof(snpCovBasisNegComps[0]));
    projMaskSnps = ALIGNED_MALLOC_UCHARS(M);
    snpData.writeMaskSnps(projMaskSnps);
    
    double *snpVector = ALIGNED_MALLOC_DOUBLES(Nstride);
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(256 * sizeof(*work));
    double lut0129[4] = {0, 1, 2, 9};    

    MprojMask = 0;
    std::set <int> projMaskChromSet;
    Xfro2 = 0;
    const vector <SnpInfo> &snps = snpData.getSnpInfo();
    for (uint64 m = 0; m < M; m++)
      if (projMaskSnps[m]) {
	// build masked snp vector with default 0129 values (note 0 can mean masked!)
	snpData.buildMaskedSnpVector(snpVector, maskIndivs, m, lut0129, work);
	projMaskSnps[m] = initMarker(m, snpVector);

	if (projMaskSnps[m]) { // may have been masked by initMarker!
	  projMaskChromSet.insert(snps[m].chrom);
	  MprojMask++;
	  Xfro2 += Xnorm2s[m];
	}
      }
    numChromsProjMask = projMaskChromSet.size();
    printf("Number of chroms with >= 1 good SNP: %d\n", numChromsProjMask);
#ifdef VERBOSE
    printf("Average norm of projected SNPs:           %f\n", Xfro2/MprojMask);
    printf("Dimension of all-1s proj space (Nused-1): %d\n", (int) (Nused-1));
    fflush(stdout);
#endif

    ALIGNED_FREE(work);
    ALIGNED_FREE(snpVector);
  }

  Bolt::~Bolt(void) {
    ALIGNED_FREE(Xnorm2s);
    ALIGNED_FREE(snpValueLookup);
    ALIGNED_FREE(snpCovBasisNegComps);
    ALIGNED_FREE(projMaskSnps);
  }

  const SnpData &Bolt::getSnpData(void) const { return snpData; }
  const CovariateBasis &Bolt::getCovBasis(void) const { return covBasis; }
  const double *Bolt::getMaskIndivs(void) const { return maskIndivs; }
  const uchar *Bolt::getProjMaskSnps(void) const { return projMaskSnps; }
  uint64 Bolt::getMprojMask(void) const { return MprojMask; }
  int Bolt::getNumChromsProjMask(void) const { return numChromsProjMask; }
  uint64 Bolt::getNused(void) const { return Nused; }
  uint64 Bolt::getNCstride(void) const { return Nstride+Cstride; }

  /**
   * pheno: needs to be copied (as it'll be extended to Nstride and projected)
   */
  Bolt::StatsDataRetroLOCO Bolt::computeLINREG(vector <double> pheno) const {

    while (pheno.size() < Nstride) pheno.push_back(0); // zero-fill to Nstride
    vector <double> stats(M, BAD_SNP_STAT);
    covBasis.applyMaskIndivs(&pheno[0]);
    covBasis.projectCovars(&pheno[0]);
    double phenoNorm2 = NumericUtils::norm2(&pheno[0], Nstride);
    NumericUtils::normalize(&pheno[0], Nstride);
    // pheno has now been projected and normalized to vector norm 1

    // note: pheno already has covariates projected out, so no need for covar comps
    double *snpVec = ALIGNED_MALLOC_DOUBLES(Nstride);
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(256 * sizeof(*work));
  
    for (uint64 m = 0; m < M; m++)
      if (projMaskSnps[m]) {
	snpData.buildMaskedSnpVector(snpVec, maskIndivs, m, snpValueLookup[m], work);
	// pheno has vector norm 1 and is orthogonal to covariates
	// ||snpVec||^2 = Xnorm2s[m]
	// dot product is really taking place in Nused-Cindep free dimensions: scale to get chisq
	stats[m] = NumericUtils::sq(NumericUtils::dot(snpVec, &pheno[0], Nstride)) / Xnorm2s[m]
	  * (Nused-Cindep);
      }
    ALIGNED_FREE(work);
    ALIGNED_FREE(snpVec);

    // save calibrated "residual" (i.e., pheno)
    double residFactor = sqrt((double) (Nused-Cindep));
    for (uint64 n = 0; n < Nstride; n++)
      pheno[n] *= residFactor;

    return StatsDataRetroLOCO("LINREG", stats, vector < vector <double> > (1, pheno),
			      computeSnpChunkEnds(vector <int> (M, 0)),
			      vector <double> (1, residFactor / sqrt(phenoNorm2)));
  }

  /**
   * computes LINREG on Bayes residual: (x_m^T phi_resid^m / (||x_m|| * ||phi_resid^m||))^2
   * calibrates by matching LDscore regression intercept to standard (inf. model) mixed model stats
   */
  Bolt::StatsDataRetroLOCO Bolt::computeLmmBayes
  (vector <double> pheno, const vector <double> &logDeltas, const vector <double> &sigma2Ks,
   double varFrac2Est, double pEst, bool MCMC, double genWindow, int physWindow, int maxIters,
   double approxLLtol, const vector <double> &statsLmmInf, const vector <double> &LDscores,
   const vector <double> &LDscoresChip) const {

    while (pheno.size() < Nstride) pheno.push_back(0); // zero-fill to Nstride
    int numLeaveOutChunks = logDeltas.size();
    double *phenoResidCovCompVecs = ALIGNED_MALLOC_DOUBLES(numLeaveOutChunks*(Nstride+Cstride));
    maskFillCovCompVecs(phenoResidCovCompVecs, &pheno[0], numLeaveOutChunks);

    /***** assign snps to LOCO chunks *****/

    vector <int> chunkAssignments = makeChunkAssignments(numLeaveOutChunks);
    uchar *batchMaskSnps = ALIGNED_MALLOC_UCHARS(M*numLeaveOutChunks);
    vector <int> chunks; for (int i = 0; i < numLeaveOutChunks; i++) chunks.push_back(i);
    vector <uint64> Mused = makeBatchMaskSnps(batchMaskSnps, chunkAssignments, chunks, genWindow,
					      physWindow);

    /***** perform Bayesian mixed model computation *****/

    double *betas = ALIGNED_MALLOC_DOUBLES(numLeaveOutChunks*M);
    vector <double> varFrac2Ests(numLeaveOutChunks, varFrac2Est);
    vector <double> pEsts(numLeaveOutChunks, pEst);
    batchComputeBayesIter(phenoResidCovCompVecs, betas, batchMaskSnps, &Mused[0], &logDeltas[0],
			  &sigma2Ks[0], &varFrac2Ests[0], &pEsts[0], numLeaveOutChunks, MCMC,
			  maxIters, approxLLtol);
    ALIGNED_FREE(betas);

    vector <double> resNorm2s(numLeaveOutChunks);
    computeProjNorm2s(&resNorm2s[0], phenoResidCovCompVecs, numLeaveOutChunks);

    /***** compute retrospective statistics *****/

    vector <double> stats(M, BAD_SNP_STAT);

    double *snpCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(256 * sizeof(*work));
    for (uint64 m = 0; m < M; m++)
      if (projMaskSnps[m]) {
	const double *phenoResidCovCompVec =
	  phenoResidCovCompVecs + chunkAssignments[m]*(Nstride+Cstride);
	
	// build snp vector + sign-flipped covar comps
	buildMaskedSnpNegCovCompVec(snpCovCompVec, m, work);
	
	// compute LINREG on Bayes residual: (x_m^T phi_resid^m / (||x_m|| * ||phi_resid^m||))^2
	double dotProd = NumericUtils::dot(snpCovCompVec, phenoResidCovCompVec, Nstride+Cstride);
	stats[m] = NumericUtils::sq(dotProd) / resNorm2s[chunkAssignments[m]] / Xnorm2s[m]
	  * (Nused-Cindep);
      }
    
    /***** calibrate statistics by matching LDscore regression intercept to inf. model stats *****/
    
    // TODO: update with final values of constants for LD Score regression
    double minMAF = 0.01;
    double outlierVarFracThresh = 0.001;
    int varianceDegree = 2;
    std::pair <double, double> calibrationFactorMeanStd =
      LDscoreCalibration::calibrateStatPair(snpData.getSnpInfo(), statsLmmInf, stats, LDscores,
					    LDscoresChip, minMAF, Nused, outlierVarFracThresh,
					    snpData.getMapAvailable(), varianceDegree);
    for (uint64 m = 0; m < M; m++)
      if (projMaskSnps[m])
	stats[m] *= calibrationFactorMeanStd.first;

    // save residuals to compute stats on non-GRM snps
    vector < vector <double> > calibratedResids(numLeaveOutChunks, vector <double> (Nstride));
    for (int i = 0; i < numLeaveOutChunks; i++) {
      double residFactor = sqrt((Nused-Cindep) / resNorm2s[i] * calibrationFactorMeanStd.first);
      for (uint64 n = 0; n < Nstride; n++)
	calibratedResids[i][n] = phenoResidCovCompVecs[i*(Nstride+Cstride) + n] * residFactor;
      covBasis.projectCovars(&calibratedResids[i][0]);
    }

    ALIGNED_FREE(work);
    ALIGNED_FREE(snpCovCompVec);
    ALIGNED_FREE(batchMaskSnps);
    ALIGNED_FREE(phenoResidCovCompVecs);

    return StatsDataRetroLOCO(MCMC ? "BOLT_LMM_MCMC" : "BOLT_LMM",
			      stats, calibratedResids, computeSnpChunkEnds(chunkAssignments));
  }

  /**
   * computes MLMe LOCO retrospective stat calibrated to prospective stat (like GRAMMAR-Gamma)
   * simultaneously solves linear systems H \ b for numerators and calibration snp denominators
   * uses previously-estimated h2 (logDeltaEst) and previously-computed Hinv*phi from VC est
   * - note that H in HinvPhi is genome-wide, not LOCO; ok for weeding outlier calibration snps
   *
   * stat: retrospective * (avg pro at random SNPs / avg retro at random SNPs)
   * retrospective stat: (N-C) * (x_m^T H_chunk[m]^-1 phi / (||x_m|| * ||H_chunk[m]^-1 phi||))^2
   * prospective stat: (N-C) * (x_m^T H_chunk[m]^-1 phi)^2
   *                   / (x_m^T H_chunk[m]^-1 x_m) / (phi^T H_chunk[m]^-1 phi)
   */
  Bolt::StatsDataRetroLOCO Bolt::computeLmmInf
  (vector <double> pheno, vector <double> logDeltas, const vector <double> &sigma2Ks,
   const double HinvPhiCovCompVec[], int numCalibSnps, double genWindow, int physWindow,
   int maxIters, double CGtol, int seed) const {
    
    while (pheno.size() < Nstride) pheno.push_back(0); // zero-fill to Nstride

    int numLeaveOutChunks = logDeltas.size();
    vector <uint64> mProSnps = selectProSnps(numCalibSnps, HinvPhiCovCompVec, seed);

    vector <int> chunkAssignments = makeChunkAssignments(numLeaveOutChunks);
    
    // batchMaskSnps[]: M x B (first numLeaveOutChunks cols are from chunk assignment
    //                         last numCalibSnps cols are copied according to chunks of pro snps)
    uint64 B = numLeaveOutChunks + numCalibSnps; // group numerator and denominator CG ops
    uchar *batchMaskSnps = ALIGNED_MALLOC_UCHARS(M*B);
    vector <int> chunks; // list of chunks correspondong to columns of batchMaskSnps
    for (int i = 0; i < numLeaveOutChunks; i++) chunks.push_back(i);
    for (int j = 0; j < numCalibSnps; j++) chunks.push_back(chunkAssignments[mProSnps[j]]);
    vector <uint64> Mused = makeBatchMaskSnps(batchMaskSnps, chunkAssignments, chunks, genWindow,
					      physWindow);

    for (int j = 0; j < numCalibSnps; j++) // append logDeltas for prospective stat snps
      logDeltas.push_back(logDeltas[chunkAssignments[mProSnps[j]]]);

    /***** compute H_chunk[i]^-1 * [phi (numLeaveOutChunks copies), proSnps x (numCalibSnps)] *****/
    // first part of batch is for all numerators; second part is for prospective stat denominators
    // todo: Hinv*phi is in HinvPhiCovCompVec; could warm-start first part if swapping of converged

    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(256 * sizeof(*work));
    double *covCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);

    // rhsCovCompVecs[]: B x (Nstride+Cstride); first rows are copies of pheno; last are pro snps)
    double *rhsCovCompVecs = ALIGNED_MALLOC_DOUBLES(B*(Nstride+Cstride));
    double *phiCovCompVecs = rhsCovCompVecs;
    double *proSnpCovCompVecs = rhsCovCompVecs + numLeaveOutChunks * (Nstride+Cstride);
    maskFillCovCompVecs(phiCovCompVecs, &pheno[0], numLeaveOutChunks);
    for (int j = 0; j < numCalibSnps; j++)
      buildMaskedSnpCovCompVec(proSnpCovCompVecs + j*(Nstride+Cstride), mProSnps[j], work);

    // HinvRhsCovCompVecs[]: B x (Nstride+Cstride); no init, but [todo] could warm-start first rows
    double *HinvRhsCovCompVecs = ALIGNED_MALLOC_DOUBLES(B*(Nstride+Cstride));
    double *HinvPhiCovCompVecs = HinvRhsCovCompVecs;
    double *HinvProSnpCovCompVecs = HinvRhsCovCompVecs + numLeaveOutChunks * (Nstride+Cstride);

    conjGradSolve(HinvRhsCovCompVecs, false, rhsCovCompVecs, batchMaskSnps, &Mused[0],
		  &logDeltas[0], B, maxIters, CGtol);
    
    // compute ||H_chunk[i]^-1 * phi||^2
    vector <double> HinvPhiNorm2s(numLeaveOutChunks);
    computeProjNorm2s(&HinvPhiNorm2s[0], HinvPhiCovCompVecs, numLeaveOutChunks);

    vector <double> phiHinvPhis(numLeaveOutChunks); // to be used in prospective stat denominators
    maskFillCovCompVecs(covCompVec, &pheno[0], 1);
    for (int i = 0; i < numLeaveOutChunks; i++)
      phiHinvPhis[i] = dotCovCompVecs(covCompVec, HinvPhiCovCompVecs + i*(Nstride+Cstride));

    // compute uncalibrated MLMe retrospective statistic (GRAMMAR = LINREG on BLUP resid):
    //   (N-C) * (x_m^T H_chunk[m]^-1 phi / (||x_m|| * ||H_chunk[m]^-1 phi||))^2
    vector <double> stats(M, BAD_SNP_STAT);
    for (uint64 m = 0; m < M; m++)
      if (projMaskSnps[m]) {
	buildMaskedSnpNegCovCompVec(covCompVec, m, work);
	int i = chunkAssignments[m];
	stats[m] =
	  NumericUtils::sq(NumericUtils::dot(HinvPhiCovCompVecs + i*(Nstride+Cstride),
					     covCompVec, Nstride+Cstride))
	  / HinvPhiNorm2s[i] / Xnorm2s[m] * (Nused-Cindep);
      }
    /*
    double phenoNorm2 = computeProjNorm2(rhsCovCompVecs);
    cout << "phenoNorm2: " << phenoNorm2 << endl;
    vector <double> resNorm2s(numLeaveOutChunks);
    for (int i = 0; i < numLeaveOutChunks; i++) {
      resNorm2s[i] = HinvPhiNorm2s[i] * exp(2*logDeltas[i]);
      cout << "resNorm2s[" << i << "]: " << resNorm2s[i] << " (" << resNorm2s[i] / phenoNorm2
	   << ")" << endl;
    }
    */

    /***** compute prospective stats at selected snps and calibrate *****/
    
    vector <double> proStats(numCalibSnps), retroStats(numCalibSnps), ratios(numCalibSnps);
    for (int j = 0; j < numCalibSnps; j++) {
      uint64 m = mProSnps[j];
      retroStats[j] = stats[m];
      buildMaskedSnpNegCovCompVec(covCompVec, m, work);
      int i = chunkAssignments[m];
      proStats[j] = // prospective stat: (N-C) * (x^T H^-1 phi)^2 / (x^T H^-1 x * phi^T H^-1 phi)
	retroStats[j] * HinvPhiNorm2s[i] * Xnorm2s[m]
	/ NumericUtils::dot(covCompVec, HinvProSnpCovCompVecs + j*(Nstride+Cstride),
			    Nstride+Cstride)
	/ phiHinvPhis[i];
      ratios[j] = proStats[j] / retroStats[j];
    }
    double totProStats = std::accumulate(proStats.begin(), proStats.end(), 0.0);
    double totRetroStats = std::accumulate(retroStats.begin(), retroStats.end(), 0.0);
    double calibrationFactor = totProStats / totRetroStats;
    vector <double> calibrationJacks(numCalibSnps);
    for (int j = 0; j < numCalibSnps; j++)
      calibrationJacks[j] = (totProStats - proStats[j]) / (totRetroStats - retroStats[j]);
    double calibrationStd = Jackknife::stddev(calibrationJacks, numCalibSnps);

    printf("\nAvgPro: %.3f   AvgRetro: %.3f   Calibration: %.3f (%.3f)   (%d SNPs)\n",
	   totProStats/numCalibSnps, totRetroStats/numCalibSnps, calibrationFactor,
	   calibrationStd, numCalibSnps);
    
    double ratioOfMedians = NumericUtils::median(proStats) / NumericUtils::median(retroStats);
    double medianOfRatios = NumericUtils::median(ratios);
    printf("Ratio of medians: %.3f   Median of ratios: %.3f\n", ratioOfMedians, medianOfRatios);

    if (calibrationStd > 0.01) {
      cerr << "WARNING: Calibration std error is high; consider increasing --numCalibSnps" << endl;
      cerr << "         Using ratio of medians instead: " << ratioOfMedians << endl;
      calibrationFactor = ratioOfMedians;
    }

    for (uint64 m = 0; m < M; m++)
      if (projMaskSnps[m])
	stats[m] *= calibrationFactor;

    // save residuals to compute stats on non-GRM snps
    vector < vector <double> > calibratedResids(numLeaveOutChunks, vector <double> (Nstride));
    vector <double> VinvScaleFactors(numLeaveOutChunks);
    for (int i = 0; i < numLeaveOutChunks; i++) {
      double residFactor = sqrt((Nused-Cindep) / HinvPhiNorm2s[i] * calibrationFactor);
      for (uint64 n = 0; n < Nstride; n++)
	calibratedResids[i][n] = HinvPhiCovCompVecs[i*(Nstride+Cstride) + n] * residFactor;
      covBasis.projectCovars(&calibratedResids[i][0]);
      VinvScaleFactors[i] = 1 / residFactor / sigma2Ks[i];
    }

    ALIGNED_FREE(HinvRhsCovCompVecs);
    ALIGNED_FREE(rhsCovCompVecs);
    ALIGNED_FREE(covCompVec);
    ALIGNED_FREE(work);
    ALIGNED_FREE(batchMaskSnps);
    
    return StatsDataRetroLOCO("BOLT_LMM_INF", stats, calibratedResids,
			      computeSnpChunkEnds(chunkAssignments), VinvScaleFactors);
  }

  /**
   * given betas, build phenotypes and collect negated covar comps
   * for out-of-sample prediction, apply covar coeffs to basis extension (to OOS indivs)
   *
   * phenoPreds: (out) B x Nstride
   * betasTrans: (in) M x B array of coefficients (or NULL for baseline pred with only cov comps)
   * fittedCovComps: (in) B x Cstride (or NULL for no cov comps)
   * extAllIndivs: 0 to apply maskIndivs, 1 to make predictions for all indivs
   */
  void Bolt::batchComputePreds(double phenoPreds[], const double betasTrans[],
			       const double fittedCovComps[], uint64 B, bool extAllIndivs) const {

    double *phenoPredNegCovCompVecs = ALIGNED_MALLOC_DOUBLES(B*(Nstride+Cstride));
    memset(phenoPredNegCovCompVecs, 0, B*(Nstride+Cstride)*sizeof(phenoPredNegCovCompVecs[0]));
    
    // if extAllIndivs, don't mask; make predictions (using cov basis ext) for all indivs
    double *noMaskIndivs = ALIGNED_MALLOC_DOUBLES(Nstride);
    for (uint64 n = 0; n < Nstride; n++) noMaskIndivs[n] = 1.0;
    const double *maskIndivsChoice = extAllIndivs ? noMaskIndivs : maskIndivs;
    const double *basisMaskChoice = covBasis.getBasis(extAllIndivs);

    double *snpNegCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(256 * sizeof(*work));
  
    if (betasTrans != NULL) {
      for (uint64 m = 0; m < M; m++) {
	if (projMaskSnps[m]) {
	  // build snp vector with mask of choice (no mask if extAllIndivs)
	  snpData.buildMaskedSnpVector(snpNegCovCompVec, maskIndivsChoice, m, snpValueLookup[m],
				       work);
	  // copy negative covar comps
	  memcpy(snpNegCovCompVec + Nstride, snpCovBasisNegComps + m*Cstride,
		 Cstride*sizeof(snpNegCovCompVec[0]));

	  // update phenoPredNegCovCompVecs: add beta * snpNegCovCompVec (DGER)
	  {
	    int M_ = Nstride+Cstride;
	    int N_ = B;
	    double ALPHA_ = 1;
	    double *X_ = snpNegCovCompVec;
	    int INCX_ = 1;
	    const double *Y_ = betasTrans + m*B; // TODO: need to be aligned?
	    int INCY_ = 1;
	    double *A_ = phenoPredNegCovCompVecs;
	    int LDA_ = Nstride+Cstride;
	    DGER_MACRO(&M_, &N_, &ALPHA_, X_, &INCX_, Y_, &INCY_, A_, &LDA_);
	  }
	}
      }
    }

    // project out covars and save output
    // could do this with DGEMM, but probably not worthwhile
    for (uint64 b = 0; b < B; b++) {
      double *phenoPredNegCovCompVec = phenoPredNegCovCompVecs + b*(Nstride+Cstride);
      for (uint64 c = 0; c < Cindep; c++) {
	double covComp = phenoPredNegCovCompVec[Nstride + c];
	if (fittedCovComps != NULL)
	  covComp += fittedCovComps[b*Cstride + c];
	for (uint64 n = 0; n < Nstride; n++)
	  phenoPredNegCovCompVec[n] += covComp * basisMaskChoice[c*Nstride + n];
      }
      memcpy(phenoPreds + b*Nstride, phenoPredNegCovCompVec, Nstride*sizeof(phenoPreds[0]));
    }

    ALIGNED_FREE(work);
    ALIGNED_FREE(snpNegCovCompVec);
    ALIGNED_FREE(noMaskIndivs);
    ALIGNED_FREE(phenoPredNegCovCompVecs);
  }

  /**
   * pheno: (in) B x Nstride
   * betasTrans: (in) M x B array of coefficients
   * predIndivs: mask; 1 for indivs to make predictions on
   */
  vector <double> Bolt::batchComputePredPVEs(double *baselinePredMSEptr, const double pheno[],
					     const double betasTrans[], uint64 B,
					     const double predIndivs[]) const {
#ifdef VERBOSE
    Timer timer;
    cout << "Computing predictions on left-out cross-validation fold" << endl;
#endif
    vector <double> MSEs(B+1); // last rep: baseline prediction using only covariates
    double numPredIndivs = NumericUtils::sum(predIndivs, Nstride);
    double *phenoPreds = ALIGNED_MALLOC_DOUBLES((B+1)*Nstride); // last rep: baseline
    double *fittedCovComps = ALIGNED_MALLOC_DOUBLES(B*Cstride);
    fillCovComps(fittedCovComps, pheno, B);
    batchComputePreds(phenoPreds + B*Nstride, NULL, fittedCovComps, 1, true);
    batchComputePreds(phenoPreds, betasTrans, fittedCovComps, B, true);
    for (uint64 b = 0; b <= B; b++) {
      for (uint64 n = 0; n < Nstride; n++)
	if (predIndivs[n])
	  MSEs[b] += NumericUtils::sq(phenoPreds[b*Nstride + n] - pheno[n]);
      MSEs[b] /= numPredIndivs;
    }
    vector <double> PVEs(B);
    for (uint64 b = 0; b < B; b++)
      PVEs[b] = 1 - MSEs[b] / MSEs[B];
    ALIGNED_FREE(fittedCovComps);
    ALIGNED_FREE(phenoPreds);
#ifdef VERBOSE
    cout << "Time for computing predictions = " << timer.update_time() << " sec" << endl;
#endif
    if (baselinePredMSEptr != NULL) *baselinePredMSEptr = MSEs[B];
    return PVEs;
  }

  void Bolt::computeWritePredBetas(const string &betasFile, vector <double> pheno, double logDelta,
				   double sigma2K, double varFrac2Est, double pEst, bool MCMC,
				   int maxIters, double approxLLtol) const {

    while (pheno.size() < Nstride) pheno.push_back(0); // zero-fill to Nstride
    double *phenoResidCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    maskFillCovCompVecs(phenoResidCovCompVec, &pheno[0], 1);

    // use all non-masked SNPs
    vector <int> chunkAssignments(M, -1);
    uchar *batchMaskSnps = ALIGNED_MALLOC_UCHARS(M);
    memset(batchMaskSnps, 0, M*sizeof(batchMaskSnps[0]));
    uint64 Mused = 0;
    for (uint64 m = 0; m < M; m++)
      if (projMaskSnps[m]) {
	chunkAssignments[m] = 0;
	batchMaskSnps[m] = 1;
	Mused++;
      }
    double *betas = ALIGNED_MALLOC_DOUBLES(M);
    batchComputeBayesIter(phenoResidCovCompVec, betas, batchMaskSnps, &Mused, &logDelta, &sigma2K,
			  &varFrac2Est, &pEst, 1, MCMC, maxIters, approxLLtol);
    
    // write betas with SNP info
    const vector <SnpInfo> &snps = snpData.getSnpInfo();

    // print header
    FileUtils::AutoGzOfstream fout; fout.openOrExit(betasFile);
    fout << std::fixed; // don't use scientific notation
    fout << "SNP" << "\t" << "CHR" << "\t" << "BP" << "\t" << "GENPOS" << "\t"
	 << "ALLELE1" << "\t" << "ALLELE0" << "\t" << "BETA" << endl;

    for (uint64 m = 0; m < M; m++) {
      const SnpInfo &snp = snps[m];
      fout << snp.ID << "\t" << snp.chrom << "\t" << snp.physpos << "\t" << snp.genpos << "\t"
	   << snp.allele1 << "\t" << snp.allele2;
      double beta = 0;
      if (projMaskSnps[m])
	beta = betas[m] * (snpValueLookup[m][1] - snpValueLookup[m][0]);
      fout << "\t" << std::setprecision(10) << beta << std::setprecision(6) << endl;
    }
    fout.close();

    // todo: also output covariate components (see batchComputePredPVEs call to fillCovComps)
    // ... or better, convert from covariate basis back to covariates

    ALIGNED_FREE(betas);
    ALIGNED_FREE(batchMaskSnps);
  }

  // convert between heritability parameterizations (approximately)
  double Bolt::logDeltaToH2(double logDelta) const {
    return Xfro2/(Xfro2 + MprojMask*(Nused-Cindep)*exp(logDelta));
  }

  // convert between heritability parameterizations (approximately)
  double Bolt::h2ToLogDelta(double h2) const {
    return log(Xfro2/(MprojMask*(Nused-Cindep)) * (1-h2)/h2);
  }

  /**
   * generates random genetic and (unscaled) environmental phenotypic component pairs
   * later, pairs will be combined as yGen + sqrt(delta)*yEnvUnscaled for MC scaling h2 estimation
   *
   * yGenCovCompVecs, yEnvUnscaledCovCompVecs: (out) B x (MCtrials+1) x Nstride+Cstride
   * - projecting out covariates: implicitly done by covComps
   * - taking into account Xnorm2s: automatically done by buildMaskedSnpVector
   * - applying maskIndivs: automatically done to snps (=> Gen component) by buildMaskedSnpVector
   *                        needs to be applied to EnvUnscaled component
   * - data layout: B batches of {MCtrials rand reps + 1 data rep}
   * - in each batch:
   *   - y..CovCompVecs[0..MCtrials-1]: Gen and EnvUnscaled components of random phenotypes
   *   - y..CovCompVecs[MCtrials]: Gen = pheno from data; EnvUnscaled = 0
   *
   * pheno: (in) real phenotype (data rep), possibly of size N or zero-filled beyond (no covComps)
   * batchMaskSnps: (in) M x B -- leave-out masks
   * Ms: (in) B -- saved values of sum(batchMaskSnps(:,b))
   */
  void Bolt::genMCscalingPhenoProjPairs
  (double yGenCovCompVecs[], double yEnvUnscaledCovCompVecs[], vector <double> pheno,
   const uchar batchMaskSnps[], const uint64 Ms[], uint64 B, int MCtrials, int seed) const {

    while (pheno.size() < Nstride+Cstride) pheno.push_back(0); // zero-fill to Nstride+Cstride
    covBasis.applyMaskIndivs(&pheno[0]);
    covBasis.computeCindepComponents(&pheno[Nstride], &pheno[0]);

    double *randnBetas = ALIGNED_MALLOC_DOUBLES(M*MCtrials);
    double *scaledMaskedBetas = ALIGNED_MALLOC_DOUBLES(M*MCtrials);
    double *randnEpsCovCompVecs = ALIGNED_MALLOC_DOUBLES(MCtrials*(Nstride+Cstride));
    memset(randnEpsCovCompVecs, 0, MCtrials*(Nstride+Cstride)*sizeof(randnEpsCovCompVecs[0]));

    boost::mt19937 rng(seed+1);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
      randn(rng, boost::normal_distribution<>(0.0, 1.0));

    // todo: revert to r232 (loop order temporarily changed to match MultiVC version for testing)
    // Gen: betas <- M randn
    for (uint64 m = 0; m < M; m++) {
      if (projMaskSnps[m]) {
	for (int t = 0; t < MCtrials; t++)
	  randnBetas[m*MCtrials + t] = randn();
      }
      else { // important! these will be multiplied by 0 but we need to ensure they aren't nan
	for (int t = 0; t < MCtrials; t++)
	  randnBetas[m*MCtrials + t] = 0;
      }
    }

    // generate MCtrials x {M randn betas (to be masked and scaled later), N randn epsilons}
    for (int t = 0; t < MCtrials; t++) {

      // EnvUnscaled: epsCovCompVec <- N randn, after the following processing...
      // - mask out maskIndivs (=> norm2 ~ Nused)
      // - compute covComps: implicitly project out covars (=> norm2-SUM(comps2) ~ Nused-Cindep)
      double *randnEpsCovCompVec = randnEpsCovCompVecs + t*(Nstride+Cstride);
      for (uint64 n = 0; n < Nstride; n++)
	if (maskIndivs[n])
	  randnEpsCovCompVec[n] = randn();
      // no need to zero out components after Cindep: already 0-initialized
      covBasis.computeCindepComponents(randnEpsCovCompVec + Nstride, randnEpsCovCompVec);

      // TODO: normalize yGenCovCompVecs to have norm exactly SUM(batchMask*invSqrtM*Xnorm2s) and epsProj to have norm exactly Nused-Cindep?      
    }

    // compute batches of actual yGen, yEnvUnscaled
    for (uint64 b = 0; b < B; b++) {
      // rand reps
      double invSqrtM = 1/sqrt((double) Ms[b]);
      // Gen: via snp coeffs: 1/sqrt(Ms[b]) * randnBetas
      for (uint64 m = 0; m < M; m++)
	for (int t = 0; t < MCtrials; t++)
	  scaledMaskedBetas[m*MCtrials + t] = (double) batchMaskSnps[m*B+b] * invSqrtM *
	    randnBetas[m*MCtrials + t];
      multX(yGenCovCompVecs + (b*(MCtrials+1))*(Nstride+Cstride), scaledMaskedBetas, MCtrials);
      // EnvUnscaled: randn masked implicitly projected epsilons[t] (same for all b; copy)
      memcpy(yEnvUnscaledCovCompVecs + (b*(MCtrials+1))*(Nstride+Cstride), randnEpsCovCompVecs,
	     MCtrials*(Nstride+Cstride)*sizeof(randnEpsCovCompVecs));

      // data rep
      // Gen: from pheno
      memcpy(yGenCovCompVecs + (b*(MCtrials+1)+MCtrials)*(Nstride+Cstride), &pheno[0],
	     (Nstride+Cstride)*sizeof(pheno[0]));
      // EnvUnscaled: 0
      memset(yEnvUnscaledCovCompVecs + (b*(MCtrials+1)+MCtrials)*(Nstride+Cstride), 0,
	     (Nstride+Cstride)*sizeof(yEnvUnscaledCovCompVecs[0]));
    }

    /*
    cout << "first batch of rand (Gen, EnvUnscaled) pheno comps + (real, 0) at end:" << endl;
    for (int t = 0; t <= MCtrials; t++) {
      cout << "genNorm2: " << computeProjNorm2(yGenCovCompVecs + t*(Nstride+Cstride)) << " ";
      cout << "envUnscaledNorm2: "
	   << computeProjNorm2(yEnvUnscaledCovCompVecs + t*(Nstride+Cstride)) << endl;
    }
    */

    ALIGNED_FREE(randnBetas);
    ALIGNED_FREE(scaledMaskedBetas);
    ALIGNED_FREE(randnEpsCovCompVecs);
  }

  /**
   * estimates f_REML at a single log(delta) for several LOCO reps at once
   *
   * testHinvPhiCovCompVec (out): Nstride+Cstride vector for later reuse
   * testVCs (in/out): contains value of log(delta) to test
   * - on return, fJacks[MCtrials+1] and fRandsAsData[MCtrials] are filled with b=0 batch values
   * - also, sigma2K is set
   * yGenCovCompVecs: (in) B x (MCtrials+1) x (Nstride+Cstride) pre-generated random yGen
   * - data layout: B batches of MCtrials rand reps followed by 1 data rep
   * yEnvUnscaledCovCompVecs: (in) B x (MCtrials+1) x (Nstride+Cstride) pre-gen random yEnvUnscaled
   * - data layout: same as above
   * batchMaskSnps: (in) M x B -- leave-out masks
   * Ms: (in) B -- saved values of sum(batchMaskSnps(:,b))
   *
   * return:
   * - B-element vector of f_REML estimates = coef2ratio(data)/coef2ratio(rand)
   *   estimated using {MCtrials rand reps + 1 data rep} for each of B batches


   * TODO: old; remove
   * - [first par] sigma2Ks: B-element vector of sigma2K variance parameters assuming delta
   */
  vector <double> Bolt::computeMCscalingFs
  (/*double sigma2Ks[], */double testHinvPhiCovCompVec[], VarCompData &testVCs/*double logDelta*/,
   const double yGenCovCompVecs[], const double yEnvUnscaledCovCompVecs[],
   const uchar batchMaskSnps[], const uint64 Ms[], uint64 B, int MCtrials, int CGmaxIters,
   double CGtol) const {

    const double logDelta = testVCs.logDelta;
#ifdef VERBOSE
    cout << "Estimating MC scaling f_REML at log(delta) = " << logDelta << ", h2 = "
	 << logDeltaToH2(logDelta) << "..." << endl;
#endif

    vector <double> MCscalingFs(B);
    // rand: scaled SUM(betahat^2), SUM(epshat^2) over all MCtrials
    vector <double> randSumBetaHat2s(B), randSumEpsHat2s(B);
    // data: scaled SUM(betahat^2), SUM(epshat^2)
    vector <double> dataSumBetaHat2s(B), dataSumEpsHat2s(B);

    double delta = exp(logDelta), sqrtDelta = sqrt(delta);

    // combine Gen and EnvUnscaled phenotypic components
    uint64 BxMCp1xNC = B*(MCtrials+1)*(Nstride+Cstride);
    double *yCombinedCovCompVecs = ALIGNED_MALLOC_DOUBLES(BxMCp1xNC);
    for (uint64 i = 0; i < BxMCp1xNC; i++)
      yCombinedCovCompVecs[i] = yGenCovCompVecs[i] + sqrtDelta * yEnvUnscaledCovCompVecs[i];

    // solve BLUP equations (all at once)
    // todo: optimize with CG warm start?
    double *HinvRhsCovCompVecs = ALIGNED_MALLOC_DOUBLES(BxMCp1xNC); // scaled residuals (epshats)
    // need to replicate data in logDeltas, masks, and Ms for conjGradSolve (each MCtrials+1 times)
    vector <double> logDeltasBxMCp1(B*(MCtrials+1), logDelta);
    uchar *batchMaskSnpsBxMCp1 = ALIGNED_MALLOC_UCHARS(M*B*(MCtrials+1));
    for (uint64 m = 0; m < M; m++)
      for (uint64 b = 0; b < B; b++) {
	uint64 iStart = (m*B + b) * (MCtrials+1);
	for (int t = 0; t <= MCtrials; t++)
	  batchMaskSnpsBxMCp1[iStart + t] = batchMaskSnps[m*B + b];
      }
    uint64 MsBxMCp1[B*(MCtrials+1)];
    for (uint64 b = 0; b < B; b++)
      for (int t = 0; t <= MCtrials; t++)
	MsBxMCp1[b*(MCtrials+1) + t] = Ms[b];

    conjGradSolve(HinvRhsCovCompVecs, false, yCombinedCovCompVecs, batchMaskSnpsBxMCp1, MsBxMCp1,
		  &logDeltasBxMCp1[0], B*(MCtrials+1), CGmaxIters, CGtol);

    // accumulate scaled SUM(epshat^2)
    vector <double> scaledEpsHat2s(B*(MCtrials+1));
    computeProjNorm2s(&scaledEpsHat2s[0], HinvRhsCovCompVecs, B*(MCtrials+1));
    for (uint64 b = 0; b < B; b++) {
      for (int t = 0; t < MCtrials; t++)
	randSumEpsHat2s[b] += scaledEpsHat2s[b*(MCtrials+1) + t];
      dataSumEpsHat2s[b] += scaledEpsHat2s[b*(MCtrials+1) + MCtrials];
    }
    
    // compute scaled BLUP coefficients (betahats) -- be careful; need to apply batchMaskSnps
    double *scaledUnmaskedBetaHats = ALIGNED_MALLOC_DOUBLES(M*B*(MCtrials+1));
    multXtrans(scaledUnmaskedBetaHats, HinvRhsCovCompVecs, B*(MCtrials+1));
    // accumulate scaled SUM(betahat^2)
    for (uint64 m = 0; m < M; m++)
      for (uint64 b = 0; b < B; b++)
	if (batchMaskSnps[m*B + b]) { // apply batchMaskSnps
	  uint64 iStart = (m*B + b) * (MCtrials+1);
	  for (int t = 0; t < MCtrials; t++)
	    randSumBetaHat2s[b] += NumericUtils::sq(scaledUnmaskedBetaHats[iStart + t]);
	  dataSumBetaHat2s[b] += NumericUtils::sq(scaledUnmaskedBetaHats[iStart + MCtrials]);
	}

    // multiply scale factors (not necessary for calculating f, but to avoid confusion)
    for (uint64 b = 0; b < B; b++) {
      double invM = 1 / (double) Ms[b];
      randSumBetaHat2s[b] *= invM; randSumEpsHat2s[b] *= NumericUtils::sq(delta);
      dataSumBetaHat2s[b] *= invM; dataSumEpsHat2s[b] *= NumericUtils::sq(delta);
    }
    /*
    cout << "randSumBetaHat2s[0]: " << randSumBetaHat2s[0] << endl;
    cout << "randSumEpsHat2s[0]: " << randSumEpsHat2s[0] << endl;
    cout << "dataSumBetaHat2s[0]: " << dataSumBetaHat2s[0] << endl;
    cout << "dataSumEpsHat2s[0]: " << dataSumEpsHat2s[0] << endl;
    */

    // form log(coef2ratio(data)/coef2ratio(rand))
    for (uint64 b = 0; b < B; b++)
      MCscalingFs[b] = log((dataSumBetaHat2s[b]/dataSumEpsHat2s[b]) /
			   (randSumBetaHat2s[b]/randSumEpsHat2s[b]));

    // compute rep data for error estimation (only for first element of batch, b=0)
    vector <double> sumBetaHat2b0Reps(MCtrials+1); // use with scaledEpsHat2s[0..MCtrials)
    for (uint64 m = 0; m < M; m++)
      if (batchMaskSnps[m*B]) {
	uint64 iStart = (m*B) * (MCtrials+1);
	for (int t = 0; t <= MCtrials; t++)
	  sumBetaHat2b0Reps[t] += NumericUtils::sq(scaledUnmaskedBetaHats[iStart + t]);
      }
    testVCs.fJacks.resize(MCtrials+1);
    for (int j = 0; j <= MCtrials; j++) {
      double dataBeta = sumBetaHat2b0Reps[MCtrials], dataEps = scaledEpsHat2s[MCtrials];
      double randBeta = 0, randEps = 0;
      for (int t = 0; t < MCtrials; t++)
	if (t != j) {
	  randBeta += sumBetaHat2b0Reps[t];
	  randEps += scaledEpsHat2s[t];
	}
      testVCs.fJacks[j] = log((dataBeta/dataEps) / (randBeta/randEps));
    }
    testVCs.fRandsAsData.resize(MCtrials);
    for (int i = 0; i < MCtrials; i++) { // use rand[i] for data rep
      double dataBeta = sumBetaHat2b0Reps[i], dataEps = scaledEpsHat2s[i];
      double randBeta = 0, randEps = 0;
      for (int t = 0; t < MCtrials; t++) { // just use all rand reps, including i, as rand reps
	randBeta += sumBetaHat2b0Reps[t];
	randEps += scaledEpsHat2s[t];
      }
      testVCs.fRandsAsData[i] = log((dataBeta/dataEps) / (randBeta/randEps));
    }

    // since we already have H_batch^-1 * y_proj (assuming the input value of delta),
    // compute variance scale parameter corresponding to delta:
    // sigma2K = y_proj'*(H_batch\y_proj) / (N-C) for each K_batch = X_batch*X_batch'/M_batch
    vector <double> sigma2Ks(B);
    for (uint64 b = 0; b < B; b++) {
      uint64 bDataBatchInd = b*(MCtrials+1) + MCtrials;
      const double *phiCovCompVec = yCombinedCovCompVecs + bDataBatchInd * (Nstride+Cstride);
      const double *HinvPhiCovCompVec = HinvRhsCovCompVecs + bDataBatchInd * (Nstride+Cstride);
      sigma2Ks[b] = dotCovCompVecs(phiCovCompVec, HinvPhiCovCompVec) / (Nused-Cindep);
      if (testHinvPhiCovCompVec != NULL) { // save HinvPhi for later use
	assert(B == 1);
	memcpy(testHinvPhiCovCompVec, HinvPhiCovCompVec,
	       (Nstride+Cstride)*sizeof(testHinvPhiCovCompVec[0]));
      }
      //printf("sigma2Ks[%d]: %f   sigma2es[%d]: %f\n",
      //       (int) b, sigma2Ks[b], (int) b, sigma2Ks[b]*delta);
    }
    testVCs.sigma2K = sigma2Ks[0];
    
    ALIGNED_FREE(yCombinedCovCompVecs); // todo: could optimize memory alloc (reuse work arrays)
    ALIGNED_FREE(HinvRhsCovCompVecs);
    ALIGNED_FREE(scaledUnmaskedBetaHats);
    ALIGNED_FREE(batchMaskSnpsBxMCp1);

    return MCscalingFs;
  }

  /**
   * estimates MC scaling f_REML at a single log(delta); updates f_REML data for best VCs
   * testVCs: (in/out)
   * - in: logDelta, stdOfReml, stdToReml (est. in calling function using secants with rep data)
   * - out: sigma2K, fJacks, fRandsAsData
   *   f_REML = fJacks.back(); others are sub/leave-out rep data for computing standard errors
   *
   * yGenCovCompVecs: (in) (MCtrials+1) x (Nstride+Cstride) pre-generated random yGen
   * yEnvUnscaledCovCompVecs: (in) (MCtrials+1) x (Nstride+Cstride) pre-gen random yEnvUnscaled

   * TODO: old; remove
   *
   * logDelta: value of log(delta) to test
   *
   * sigma2Kbest: (in/out) current sigma2K corresponding to best logDelta, updated if new best
   * logDeltaBest: (in/out) current argmin of |f_REML| estimate, updated if bested
   * bestAbsF: (in/out) current minimal |f_REML| estimate, updated if bested
   *
   * return: f_REML estimate = coef2ratio(data)/coef2ratio(rand) using MCtrials rand + 1 data rep
   */
  void Bolt::updateBestMCscalingF(VarCompData &bestVCs,
//double *sigma2Kbest, double *logDeltaBest, double *bestAbsF,
					double HinvPhiCovCompVec[], VarCompData &testVCs,
				    const double yGenCovCompVecs[],
				    const double yEnvUnscaledCovCompVecs[], int MCtrials,
				    int CGmaxIters, double CGtol) const {

    double *testHinvPhiCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    computeMCscalingFs(/*&testVCs.sigma2K, */testHinvPhiCovCompVec, testVCs, yGenCovCompVecs,
			 yEnvUnscaledCovCompVecs, projMaskSnps, &MprojMask, 1, MCtrials,
			 CGmaxIters, CGtol);
#ifdef VERBOSE
    printf("  MCscaling: logDelta = %.2f, h2 = %.3f, f = %g\n\n", testVCs.logDelta,
	   logDeltaToH2(testVCs.logDelta), testVCs.fJacks.back());
#endif
    if (bestVCs.fJacks.empty() || fabs(testVCs.fJacks.back()) < fabs(bestVCs.fJacks.back())) {
      bestVCs = testVCs;
      memcpy(HinvPhiCovCompVec, testHinvPhiCovCompVec,
	     (Nstride+Cstride)*sizeof(HinvPhiCovCompVec[0]));
    }
    ALIGNED_FREE(testHinvPhiCovCompVec);
  }
  
  /**
   * re-estimates log(delta) for several LOCO reps at once, given the all-SNPs estimate logDeltaEst
   * calculates estimates using secants between logDeltaEst and logDeltaEst + log(B/B-2))
   * upper-bounds h2 (i.e., lower-bounds delta) with all-SNPs estimate
   *
   * pheno: (in) real phenotype, possibly of size N or zero-filled beyond (no covComps)
   * logDeltaEst: starting estimate of log(delta), presumably from all-SNPs (no LOCO) run
   * batchMaskSnps: (in) M x B -- leave-out masks
   * Ms: (in) B -- saved values of sum(batchMaskSnps(:,b))
   *
   * return: B-element vector of log(delta) estimates for each LOCO rep
   */
  vector <double> Bolt::reEstLogDeltas(const vector <double> &pheno, double logDeltaEst,
				       const uchar batchMaskSnps[], const uint64 Ms[], uint64 B,
				       int MCtrials, int CGmaxIters, double CGtol, int seed)
    const {

    cout << endl << "=== Re-estimating variance parameters for " << B << " leave-out reps ==="
	 << endl << endl;

    setMCtrials(MCtrials);

    vector <double> logDeltaReEsts(B);

    double *yGenCovCompVecs = ALIGNED_MALLOC_DOUBLES(B * (MCtrials+1) * (Nstride+Cstride));
    double *yEnvUnscaledCovCompVecs = ALIGNED_MALLOC_DOUBLES(B * (MCtrials+1) * (Nstride+Cstride));

    genMCscalingPhenoProjPairs(yGenCovCompVecs, yEnvUnscaledCovCompVecs, pheno, batchMaskSnps,
			       Ms, B, MCtrials, seed);

    double logDeltaTests[2];
    logDeltaTests[0] = logDeltaEst;
    logDeltaTests[1] = logDeltaEst + (B > 2 ? log(B/(B-2.0)) : 1);
    vector < vector <double> > MCscalingFsBoth(2);
    //double sigma2Ks[B]; // unused TODO: remove
    // todo: can optimize further by computing both logDelta tests with one batched CG
    for (int i = 0; i < 2; i++) {
      VarCompData testVCs; testVCs.logDelta = logDeltaTests[i];
      MCscalingFsBoth[i] = computeMCscalingFs(/*sigma2Ks, */NULL, testVCs, yGenCovCompVecs,
					      yEnvUnscaledCovCompVecs, batchMaskSnps, Ms, B,
					      MCtrials, CGmaxIters, CGtol);
    }
    for (uint64 b = 0; b < B; b++) {
      // interp (logDeltaTests[0], MCscalingFsBoth[0][b])-(logDeltaTests[1], MCscalingFsBoth[1][b])
      double xPrev = logDeltaTests[0], fPrev = MCscalingFsBoth[0][b];
      double xCur = logDeltaTests[1], fCur = MCscalingFsBoth[1][b];
      logDeltaReEsts[b] = (xPrev*fCur - xCur*fPrev) / (fCur - fPrev);
      if (std::isnan(logDeltaReEsts[b]) || logDeltaReEsts[b] < logDeltaEst) {
	cerr << "WARNING: Estimated h2 on leave-out batch " << b << " exceeds all-SNPs h2" << endl;
	cerr << "         Replacing " << logDeltaToH2(logDeltaReEsts[b]) << " with "
	     << logDeltaToH2(logDeltaEst) << endl;
	logDeltaReEsts[b] = logDeltaEst; // TODO: is it safest to bound LOCO h2 with all-SNPs h2?
      }
#ifdef VERBOSE
      printf("MCscaling:   logDelta[%d] = %f,   h2 = %.3f,   Mused = %d  (%4.1f%%)\n",
	     (int) b, logDeltaReEsts[b], logDeltaToH2(logDeltaReEsts[b]), (int) Ms[b],
	     (double) Ms[b] / MprojMask * 100);
#endif
    }
    cout << endl;

    ALIGNED_FREE(yGenCovCompVecs);
    ALIGNED_FREE(yEnvUnscaledCovCompVecs);
    
    return logDeltaReEsts;
  }

  /**
   * (in/out): logDeltas, sigma2Ks
   */
  void Bolt::reEstVCs(vector <double> pheno, vector <double> &logDeltas, vector <double> &sigma2Ks,
		      int reEstMCtrials, double genWindow, int physWindow, int maxIters,
		      double CGtol, int seed) const {

    while (pheno.size() < Nstride) pheno.push_back(0); // zero-fill to Nstride
    int numLeaveOutChunks = logDeltas.size();
    double *phenoResidCovCompVecs = ALIGNED_MALLOC_DOUBLES(numLeaveOutChunks*(Nstride+Cstride));
    maskFillCovCompVecs(phenoResidCovCompVecs, &pheno[0], numLeaveOutChunks);

    /***** assign snps to LOCO chunks *****/

    vector <int> chunkAssignments = makeChunkAssignments(numLeaveOutChunks);
    uchar *batchMaskSnps = ALIGNED_MALLOC_UCHARS(M*numLeaveOutChunks);
    vector <int> chunks; for (int i = 0; i < numLeaveOutChunks; i++) chunks.push_back(i);
    vector <uint64> Mused = makeBatchMaskSnps(batchMaskSnps, chunkAssignments, chunks, genWindow,
					      physWindow);

    /**** set variance parameters for LOCO chunks *****/

    logDeltas = reEstLogDeltas(pheno, logDeltas[0], batchMaskSnps, &Mused[0],
			       numLeaveOutChunks, reEstMCtrials, maxIters, CGtol, seed);
    // re-estimate variance scale parameter for each LOCO rep:
    // compute sigma2K = y_proj'*(H\y_proj) / (N-C)   (same CG computation as above!)
    double *HinvPhiCovCompVecs = ALIGNED_MALLOC_DOUBLES(numLeaveOutChunks*(Nstride+Cstride));
    conjGradSolve(HinvPhiCovCompVecs, false, phenoResidCovCompVecs, batchMaskSnps, &Mused[0],
		  &logDeltas[0], numLeaveOutChunks, maxIters, CGtol);

    for (int b = 0; b < numLeaveOutChunks; b++) {
      const double *phiCovCompVec = phenoResidCovCompVecs + b * (Nstride+Cstride);
      const double *HinvPhiCovCompVec = HinvPhiCovCompVecs + b * (Nstride+Cstride);
      sigma2Ks[b] = dotCovCompVecs(phiCovCompVec, HinvPhiCovCompVec) / (Nused-Cindep);
#ifdef VERBOSE
      printf("sigma2Ks[%d]: %f   sigma2es[%d]: %f\n",
	     (int) b, sigma2Ks[b], (int) b, exp(logDeltas[b]) * sigma2Ks[b]);
#endif
    }
    ALIGNED_FREE(HinvPhiCovCompVecs);
    ALIGNED_FREE(phenoResidCovCompVecs);
  }

  void Bolt::setMCtrials(int &MCtrials) const {
    if (MCtrials <= 0) {
      MCtrials = std::max(std::min((int) (4e9/Nused/Nused), 15), 3);
      cout << "Using default number of random trials: " << MCtrials
	   << " (for Nused = " << Nused << ")" << endl << endl;
    }
    else {
      cout << "Using " << MCtrials << " random trials" << endl << endl;
    }
  }

  /**
   * estimates log(delta) using all SNPs using secant method on MC scaling f_REML curve
   * also returns corresponding sigma2K and Hinv*phi (H using estimated log(delta)) for later use
   *
   * sigma2Kbest: (out) variance parameter for kinship (GRM) component
   * HinvPhiCovCompVec: (out) (Nstride+Cstride)-vector, allocated with aligned memory
   * pheno: (in) real phenotype, possibly of size N or zero-filled beyond (no covComps)
   */
  double Bolt::estLogDelta(double *sigma2Kbest, double HinvPhiCovCompVec[],
			   const vector <double> &pheno, int MCtrials,
			   double logDeltaTol, int CGmaxIters, double CGtol, int seed,
			   bool allowh2g01) const {

    setMCtrials(MCtrials);

    //double logDeltaBest = 0; double bestAbsF = 1e9;
    const double MAX_ABS_LOG_DELTA = 10;

    double *yGenCovCompVecs = ALIGNED_MALLOC_DOUBLES((MCtrials+1) * (Nstride+Cstride));
    double *yEnvUnscaledCovCompVecs = ALIGNED_MALLOC_DOUBLES((MCtrials+1) * (Nstride+Cstride));

    genMCscalingPhenoProjPairs(yGenCovCompVecs, yEnvUnscaledCovCompVecs, pheno, projMaskSnps,
			       &MprojMask, 1, MCtrials, seed);

    VarCompData bestVCs;
    
    VarCompData prevVCs; prevVCs.logDelta = h2ToLogDelta(*sigma2Kbest);
    updateBestMCscalingF(bestVCs/*sigma2Kbest, &logDeltaBest, &bestAbsF*/, HinvPhiCovCompVec,
					prevVCs/*xPrev*/, yGenCovCompVecs, yEnvUnscaledCovCompVecs, MCtrials,
					CGmaxIters, CGtol);
    
    VarCompData curVCs; curVCs.logDelta = h2ToLogDelta(prevVCs.fJacks.back() < 0 ?
						       (*sigma2Kbest)/2 : (*sigma2Kbest)*2);
    updateBestMCscalingF(bestVCs/*sigma2Kbest, &logDeltaBest, &bestAbsF*/, HinvPhiCovCompVec,
				       curVCs/*xCur*/, yGenCovCompVecs, yEnvUnscaledCovCompVecs, MCtrials,
				       CGmaxIters, CGtol);

    if (fabs(prevVCs.fJacks.back()) < fabs(curVCs.fJacks.back()))
      std::swap(prevVCs, curVCs);

    bestVCs.fJacks.clear(); // forces bestVCs to be taken from secant iters below to get std errors

    // secant iteration
    const int maxSecantIters = 5; bool converged = false;
    for (int s = 0; s < maxSecantIters; s++) {
      double xPrev = prevVCs.logDelta, xCur = curVCs.logDelta;
      vector <double> xNextJacks(MCtrials+1), xNextRandsAsData(MCtrials);
      for (int t = 0; t <= MCtrials; t++) {
	double fPrev = prevVCs.fJacks[t], fCur = curVCs.fJacks[t];
	xNextJacks[t] = (xPrev*fCur - xCur*fPrev) / (fCur - fPrev);
      }
      VarCompData nextVCs;
      nextVCs.logDelta = xNextJacks.back();
      if (!(nextVCs.logDelta > -MAX_ABS_LOG_DELTA)) nextVCs.logDelta = -MAX_ABS_LOG_DELTA;
      if (!(nextVCs.logDelta < MAX_ABS_LOG_DELTA)) nextVCs.logDelta = MAX_ABS_LOG_DELTA;
      for (int t = 0; t < MCtrials; t++) {
	/* wrong: rands can't be used as data because they're *different* for different logDeltas
	double fPrev = prevVCs.fRandsAsData[t], fCur = curVCs.fRandsAsData[t];
	xNextRandsAsData[t] = (xPrev*fCur - xCur*fPrev) / (fCur - fPrev);
	*/
	// instead, just model error in fCur by substituting each rand rep for data rep
	// assume slope of f is roughly correct
	double fPrev = prevVCs.fJacks.back(), fCur = curVCs.fJacks.back();
	xNextRandsAsData[t] = nextVCs.logDelta
	  - (curVCs.fRandsAsData[t] - fCur) * (xCur - xPrev) / (fCur - fPrev);

	// convert to h2 scale for std err display
	xNextJacks[t] = logDeltaToH2(xNextJacks[t]);
	xNextRandsAsData[t] = logDeltaToH2(xNextRandsAsData[t]);
      }
      nextVCs.stdOfReml = StatsUtils::stdDev(xNextRandsAsData);
      nextVCs.stdToReml = Jackknife::stddev(xNextJacks, MCtrials);
      
      // check exit condition
      if (bestVCs.logDelta == curVCs.logDelta
	  && fabs(nextVCs.logDelta - curVCs.logDelta) < logDeltaTol) {
	cout << "Secant iteration for h2 estimation converged in " << s << " steps" << endl;
	converged = true;
	break;
      }
      
      prevVCs = curVCs;
      curVCs = nextVCs;
      updateBestMCscalingF(bestVCs/*sigma2Kbest, &logDeltaBest, &bestAbsF*/, HinvPhiCovCompVec,
				  curVCs/*xCur*/,
				  yGenCovCompVecs, yEnvUnscaledCovCompVecs, MCtrials, CGmaxIters,
				  CGtol);
    }
    if (!converged)
      cerr << "WARNING: Secant iteration for h2 estimation may not have converged" << endl;

    /*
    printf("Estimated (pseudo-)heritability: h2 = %.3f (%.3f)\n", logDeltaToH2(bestVCs.logDelta),
	   bestVCs.stdOfReml);
    printf("Estimated std err of stochastic estimate vs. standard REML: %.3f\n",
	   bestVCs.stdToReml);
    if (bestVCs.stdToReml > 0.01) {
      cerr << "NOTE: An estimate closer to REML can be obtained with higher --h2EstMCtrials"
	   << endl;
      cerr << "      However, the underlying uncertainty of h2 is governed by sample size" << endl;
    }
    */
    printf("Estimated (pseudo-)heritability: h2g = %.3f\n", logDeltaToH2(bestVCs.logDelta));
    cout << "To more precisely estimate variance parameters and estimate s.e., use --reml" << endl;

#ifdef VERBOSE
    printf("Variance params: sigma^2_K = %f, logDelta = %f, f = %g\n",
	   bestVCs.sigma2K, bestVCs.logDelta, bestVCs.fJacks.back()/**sigma2Kbest, logDeltaBest, bestAbsF*/);
#endif
    cout << endl;

    if (fabs(bestVCs.logDelta) == MAX_ABS_LOG_DELTA) {
      if (!allowh2g01) {
	if (bestVCs.logDelta > 0)
	  cerr << "ERROR: Heritability estimate is close to 0; LMM may not correct confounding\n"
	       << "       Instead, use PC-corrected linear/logistic regression on unrelateds"
	       << endl;
	else
	  cerr << "ERROR: Heritability estimate is close to 1; algorithm may not converge\n"
	       << "       Analysis may be unsuitable due to low sample size or case ascertainment"
	       << endl;
	exit(1);
      }
      else if (bestVCs.logDelta > 0)
	cerr << "WARNING: Heritability estimate is close to 0; LMM may not correct confounding"
	     << endl;
      else
	cerr << "WARNING: Heritability estimate is close to 1; algorithm may not converge" << endl;
    }

    ALIGNED_FREE(yGenCovCompVecs);
    ALIGNED_FREE(yEnvUnscaledCovCompVecs);

    *sigma2Kbest = bestVCs.sigma2K;
    return bestVCs.logDelta;//logDeltaBest;
  }

  void Bolt::printStatsHeader(FileUtils::AutoGzOfstream &fout, bool verboseStats, bool info,
			      const vector <StatsDataRetroLOCO> &retroData) const {
    fout << std::fixed; // don't use scientific notation
    fout << "SNP" << "\t" << "CHR" << "\t" << "BP" << "\t" << "GENPOS" << "\t"
	 << "ALLELE1" << "\t" << "ALLELE0" << "\t" << "A1FREQ";
    if (info) fout << "\t" << "INFO";
    else fout << "\t" << "F_MISS";
    bool beta_printed = false;
    for (uint64 s = 0; s < retroData.size(); s++) {
      if (retroData[s].VinvScaleFactors.size() > 1U) { // infinitesimal model stat: approx beta, se
	fout << "\t" << "BETA" << "\t" << "SE";
	beta_printed = true;
      }
      if (verboseStats) // only output chisq if verbose output
	fout << "\t" << "CHISQ_" << retroData[s].statName;
      fout << "\t" << "P_" << retroData[s].statName; // always output p-value
    }
    if (!beta_printed) // special case: linear regression only
      fout << "\t" << "BETA" << "\t" << "SE";
    fout << endl;
  }

  string Bolt::getSnpStats(const string &ID, int chrom,
			   int physpos, double genpos, const string &allele1,
			   const string &allele0, double alleleFreq, double missing,
			   double workVec[], bool verboseStats,
			   const vector <StatsDataRetroLOCO> &retroData, double info) const {

    std::ostringstream fout;
    // output snp info
    fout << ID << "\t" << chrom << "\t" << physpos << "\t" << genpos << "\t"
	 << allele1 << "\t" << allele0 << "\t" << alleleFreq;

    if (info != -9) fout << "\t" << info;
    else fout << "\t" << missing;

    // compute components along cov basis vectors and put in [Nstride..Nstride+Cstride)
    // no need to zero out components after Cindep: workVec already 0-initialized
    covBasis.computeCindepComponents(workVec + Nstride, workVec);
    double projNorm2 = computeProjNorm2(workVec);
    double invNorm2 = 1 / projNorm2;
    double dotProd, stat;

    boost::math::chi_squared chisq_dist(1);

    // compute and output assoc stats
    bool beta_printed = false;
    for (uint s = 0; s < retroData.size(); s++) {
      int chunk = findChunkAssignment(retroData[s].snpChunkEnds, chrom, physpos);

      const vector <double> &calibratedResidsChunk = retroData[s].calibratedResids[chunk];
      dotProd = NumericUtils::dot(workVec, &calibratedResidsChunk[0], Nstride);
      stat = BAD_SNP_STAT; double pValue = 1;
      if (!(projNorm2 < 0.1)) {
	stat = invNorm2 * NumericUtils::sq(dotProd);
	pValue = boost::math::cdf(complement(chisq_dist, stat));
	//if (dotProd<0) stat = -stat;
      }
      char pValueBuf[100];
      if (pValue != 0)
	sprintf(pValueBuf, "%.1E", pValue);
      else {
	double log10p = log10(2.0) - M_LOG10E*stat/2 - 0.5*log10(stat*2*M_PI);
	int exponent = floor(log10p);
	double fraction = pow(10.0, log10p - exponent);
	if (fraction >= 9.95) {
	  fraction = 1;
	  exponent++;
	}
	sprintf(pValueBuf, "%.1fE%d", fraction, exponent);
      }

      if (retroData[s].VinvScaleFactors.size() > 1U) { // infinitesimal model: approx beta, se
	double xPerAlleleVinvPhi = retroData[s].VinvScaleFactors[chunk] * dotProd;
	double beta = (stat==BAD_SNP_STAT) ? 0 : stat / xPerAlleleVinvPhi;
	double se = fabs(beta) / sqrt(stat);
	fout << "\t" << beta << "\t" << se;
	beta_printed = true;
      }
      if (verboseStats) // only output chisq if verbose output
	fout << "\t" << stat;
      fout << "\t" << string(pValueBuf); // always output p-value
    }
    if (!beta_printed) { // special case: linear regression only
      double beta = (stat==BAD_SNP_STAT) ? 0 : invNorm2*dotProd / retroData[0].VinvScaleFactors[0];
      double se = fabs(beta) / sqrt(stat);
      fout << "\t" << beta << "\t" << se;	
    }
    fout << endl;
    return fout.str();
  }

  string Bolt::getSnpStats(const string &ID, int chrom,
			   int physpos, double genpos, const string &allele1,
			   const string &allele0, const uchar genoLine[], bool verboseStats,
			   const vector <StatsDataRetroLOCO> &retroData, double workVec[],
			   double info) const {

    double alleleFreq = snpData.computeAlleleFreq(genoLine, maskIndivs);
    double missing = snpData.computeSnpMissing(genoLine, maskIndivs);
    snpData.genoLineToMaskedSnpVector(workVec, genoLine, maskIndivs, alleleFreq);
    
    return getSnpStats(ID, chrom, physpos, genpos, allele1, allele0, alleleFreq, missing, workVec,
		       verboseStats, retroData, info);
  }
  string Bolt::getSnpStats(const string &ID, int chrom,
			   int physpos, double genpos, const string &allele1,
			   const string &allele0, double dosageLine[], bool verboseStats,
			   const vector <StatsDataRetroLOCO> &retroData, double info) const {

    double alleleFreq = snpData.computeAlleleFreq(dosageLine, maskIndivs);
    double missing = snpData.computeSnpMissing(dosageLine, maskIndivs);
    snpData.dosageLineToMaskedSnpVector(dosageLine, maskIndivs, alleleFreq);
    
    return getSnpStats(ID, chrom, physpos, genpos, allele1, allele0, alleleFreq, missing,
		       dosageLine, verboseStats, retroData, info);
  }

  /**
   * compute retrospective LOCO assoc stats at all SNPs in input files
   * retroData contains calibrated residuals s.t.:
   *     stat for snp x = dot(x / projNorm(x), resid[chunk])^2
   * streams genotypes from input files and streams stats to output file
   */
  void Bolt::streamComputeRetroLOCO
  (const string &outFile, const vector <string> &bimFiles, const vector <string> &bedFiles,
   const string &geneticMapFile, bool verboseStats,
   const vector <StatsDataRetroLOCO> &retroData) const {

    FileUtils::AutoGzOfstream fout; fout.openOrExit(outFile);
    
    printStatsHeader(fout, verboseStats, false, retroData);

    MapInterpolater mapInterpolater(geneticMapFile);
    const vector <int> &bedSnpToGrmIndex = snpData.getBedSnpToGrmIndex();

    FileUtils::AutoGzIfstream finBim, finBed;
    uint64 mbed = 0;
    uchar *genoLine = ALIGNED_MALLOC_UCHARS(Nstride);
    uchar *bedLineIn = ALIGNED_MALLOC_UCHARS((snpData.getNbed()+3)>>2);
    double *snpCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    memset(snpCovCompVec, 0, (Nstride+Cstride)*sizeof(snpCovCompVec[0])); // important!

    for (uint f = 0; f < bimFiles.size(); f++) {
      finBim.openOrExit(bimFiles[f]);
      finBed.openOrExit(bedFiles[f], std::ios::in | std::ios::binary);
      finBed.read((char *) genoLine, 3); // header
      string line;
      while (getline(finBim, line)) {
	// read bim info
	std::istringstream iss(line);
	string chromStr, ID, allele1, allele0; double genpos; int physpos;
	iss >> chromStr >> ID >> genpos >> physpos >> allele1 >> allele0;
	int chrom = SnpData::chrStrToInt(chromStr, Nautosomes);
	if (!geneticMapFile.empty())
	  genpos = mapInterpolater.interp(chrom, physpos);
	
	// read bed genotypes
	snpData.readBedLine(genoLine, bedLineIn, finBed, true);

	if (bedSnpToGrmIndex[mbed] != -2) // not excluded	  
	  fout << getSnpStats(ID, chrom, physpos, genpos, allele1, allele0, genoLine, verboseStats,
			      retroData, snpCovCompVec);
	mbed++;
      }
      finBed.close();
      finBim.close();
    }

    ALIGNED_FREE(snpCovCompVec);
    ALIGNED_FREE(bedLineIn);
    ALIGNED_FREE(genoLine);
    fout.close();
  }

  void Bolt::streamDosages
  (const string &outFile, const vector <string> &dosageFiles, const string &dosageFidIidFile,
   const string &geneticMapFile, bool verboseStats,
   const vector <StatsDataRetroLOCO> &retroData) const {

    FileUtils::AutoGzOfstream fout; fout.openOrExit(outFile);
    
    printStatsHeader(fout, verboseStats, false, retroData);

    MapInterpolater mapInterpolater(geneticMapFile); // ok if no map; then always returns 0
    vector < std::pair <string, string> > dosageIDs = FileUtils::readFidIids(dosageFidIidFile);
    uint Ndosage = dosageIDs.size();
    vector <uint64> dosageIndivInds(Ndosage);
    for (uint i = 0; i < Ndosage; i++)
      dosageIndivInds[i] = snpData.getIndivInd(dosageIDs[i].first, dosageIDs[i].second);

    FileUtils::AutoGzIfstream finDosage;
    double *snpCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    memset(snpCovCompVec+Nstride, 0, Cstride * sizeof(snpCovCompVec[0])); // important!
    for (uint n = 0; n < Nstride; n++) snpCovCompVec[n] = -9; // initalize dosages to missing

    for (uint f = 0; f < dosageFiles.size(); f++) {
      finDosage.openOrExit(dosageFiles[f]);
      int lineNum = 1;
      string line;
      while (getline(finDosage, line)) {
	// read snp info
	std::istringstream iss(line);
	string ID, chromStr, allele1, allele0; int physpos;
	iss >> ID >> chromStr >> physpos >> allele1 >> allele0;
	int chrom = SnpData::chrStrToInt(chromStr, Nautosomes);
	double genpos = mapInterpolater.interp(chrom, physpos); // 0 if no map
	
	// read genotype dosages
	for (uint i = 0; i < Ndosage; i++) {
	  double dosage; iss >> dosage;
	  if (dosageIndivInds[i] != SnpData::IND_MISSING)
	    snpCovCompVec[dosageIndivInds[i]] = dosage;
	}
	if (!iss) {
	  cerr << "ERROR: Too few fields in line " << lineNum << " of " << dosageFiles[f] << endl;
	  exit(1);
	}
	if (iss >> line) {
	  cerr << "ERROR: Too many fields in line " << lineNum << " of " << dosageFiles[f] << endl;
	  exit(1);
	}

	fout << getSnpStats(ID, chrom, physpos, genpos, allele1, allele0, snpCovCompVec,
			    verboseStats, retroData);
	lineNum++;
      }
      finDosage.close();
    }

    ALIGNED_FREE(snpCovCompVec);
    fout.close();
  }

  void Bolt::streamDosage2
  (const string &outFile, const vector <string> &dosage2MapFiles,
   const vector <string> &dosage2GenoFiles, bool verboseStats,
   const vector <StatsDataRetroLOCO> &retroData) const {

    FileUtils::AutoGzOfstream fout; fout.openOrExit(outFile);
    
    printStatsHeader(fout, verboseStats, false, retroData);

    FileUtils::AutoGzIfstream finMap, finGeno;
    double *snpCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    memset(snpCovCompVec+Nstride, 0, Cstride * sizeof(snpCovCompVec[0])); // important!
    for (uint n = 0; n < Nstride; n++) snpCovCompVec[n] = -9; // initalize dosages to missing

    for (uint f = 0; f < dosage2MapFiles.size(); f++) {

      finMap.openOrExit(dosage2MapFiles[f]);
      finGeno.openOrExit(dosage2GenoFiles[f]);
      vector < std::pair <string, string> > dosage2IDs;
      
      string mapLine, genoLine; getline(finGeno, genoLine);
      std::istringstream issGenoHeader(genoLine);
      string SNP, A1, A2, FID, IID; issGenoHeader >> SNP >> A1 >> A2;
      while (issGenoHeader >> FID >> IID) dosage2IDs.push_back(std::make_pair(FID, IID));     
      uint Ndosage2 = dosage2IDs.size();
      vector <uint64> dosage2IndivInds(Ndosage2);
      for (uint i = 0; i < Ndosage2; i++)
	dosage2IndivInds[i] = snpData.getIndivInd(dosage2IDs[i].first, dosage2IDs[i].second);
      
      int lineNum = 1;
      while (getline(finMap, mapLine) && getline(finGeno, genoLine)) {
	// read snp info
	std::istringstream issMapLine(mapLine);
	string chromStr, rsID; double genpos; int physpos;
	issMapLine >> chromStr >> rsID >> genpos >> physpos;
	int chrom = SnpData::chrStrToInt(chromStr, Nautosomes);

	std::istringstream issGenoLine(genoLine);
	string snpID, allele1, allele0;
	issGenoLine >> snpID >> allele1 >> allele0;
	if (snpID != rsID) {
	  cerr << "ERROR: SNP ID of line " << lineNum
	       << " of genotype file (after header) does not match map file" << endl;
	  cerr << "       dosage2 map file: " << dosage2MapFiles[f] << endl;
	  cerr << "       dosage2 genotype file: " << dosage2GenoFiles[f] << endl;
	  exit(1);
	}
	
	// read genotype probabilities
	for (uint i = 0; i < Ndosage2; i++) {
	  double p11, p10;
	  issGenoLine >> p11 >> p10;
	  double dosage = 2*p11 + p10;
	  if (dosage2IndivInds[i] != SnpData::IND_MISSING)
	    snpCovCompVec[dosage2IndivInds[i]] = dosage;
	}
	if (!issGenoLine) {
	  cerr << "ERROR: Too few fields in line " << lineNum << " of " << dosage2GenoFiles[f]
	       << endl;
	  exit(1);
	}
	if (issGenoLine >> genoLine) {
	  cerr << "ERROR: Too many fields in line " << lineNum << " of " << dosage2GenoFiles[f]
	       << endl;
	  exit(1);
	}

	fout << getSnpStats(rsID, chrom, physpos, genpos, allele1, allele0, snpCovCompVec,
			    verboseStats, retroData);
	lineNum++;
      }
      finGeno.close();
      finMap.close();
    }

    ALIGNED_FREE(snpCovCompVec);
    fout.close();
  }

  void Bolt::streamImpute2
  (const string &outFile, const vector <string> &impute2Files, const vector <int> &impute2Chroms,
   const string &impute2FidIidFile, double impute2MinMAF, const string &geneticMapFile,
   bool verboseStats, const vector <StatsDataRetroLOCO> &retroData) const {

    const double impute2CallThresh = 0.95; // seems to have no effect; IMPUTE2 always has total p=1
    FileUtils::AutoGzOfstream fout; fout.openOrExit(outFile);
    
    printStatsHeader(fout, verboseStats, true, retroData);

    MapInterpolater mapInterpolater(geneticMapFile); // ok if no map; then always returns 0
    vector < std::pair <string, string> > impute2IDs = FileUtils::readFidIids(impute2FidIidFile);
    uint Nimpute2 = impute2IDs.size();
    vector <uint64> impute2IndivInds(Nimpute2);
    int numFound = 0;
    for (uint i = 0; i < Nimpute2; i++) {
      impute2IndivInds[i] = snpData.getIndivInd(impute2IDs[i].first, impute2IDs[i].second);
      if (impute2IndivInds[i] != SnpData::IND_MISSING) numFound++;
    }
    cout << endl << "Read " << Nimpute2 << " indivs; using "
	 << numFound << " in filtered PLINK data" << endl;

    FileUtils::AutoGzIfstream finImpute2;
    double *snpCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    memset(snpCovCompVec+Nstride, 0, Cstride * sizeof(snpCovCompVec[0])); // important!
    for (uint n = 0; n < Nstride; n++) snpCovCompVec[n] = -9; // initalize dosages to missing

    for (uint f = 0; f < impute2Files.size(); f++) {
      finImpute2.openOrExit(impute2Files[f]);
      int lineNum = 1;
      string line;
      while (getline(finImpute2, line)) {
	// read snp info
	std::istringstream iss(line);
	string snpID, rsID, allele1, allele0; int physpos;
	iss >> snpID >> rsID >> physpos >> allele1 >> allele0;
	int chrom = impute2Chroms[f];
	double genpos = mapInterpolater.interp(chrom, physpos); // 0 if no map
	
	double Ncalled = 0; double sum_eij = 0, sum_fij_minus_eij2 = 0; // for INFO
	// read genotype probabilities
	for (uint i = 0; i < Nimpute2; i++) {
	  double p11, p10, p00;
	  iss >> p11 >> p10 >> p00;
	  double pTot = p11 + p10 + p00;
	  double dosage = pTot >= impute2CallThresh ? (2*p11 + p10) / pTot : -9;
	  if (pTot >= impute2CallThresh) { // for INFO
	    double eij = dosage;
	    double fij = (4*p11 + p10) / pTot;
	    sum_eij += eij;
	    sum_fij_minus_eij2 += fij - eij*eij;
	    Ncalled++;
	  }
	  if (impute2IndivInds[i] != SnpData::IND_MISSING)
	    snpCovCompVec[impute2IndivInds[i]] = dosage;
	}
	if (!iss) {
	  cerr << "ERROR: Too few fields in line " << lineNum << " of " << impute2Files[f] << endl;
	  exit(1);
	}
	if (iss >> line) {
	  cerr << "ERROR: Too many fields in line " << lineNum << " of " << impute2Files[f]
	       << endl;
	  exit(1);
	}

	double thetaHat = sum_eij / (2*Ncalled);
	double info = (thetaHat==0 || thetaHat==1) ? 1 :
	  1 - sum_fij_minus_eij2 / (2*Ncalled*thetaHat*(1-thetaHat));
	if (std::min(thetaHat, 1-thetaHat) >= impute2MinMAF)
	  fout << getSnpStats(rsID, chrom, physpos, genpos, allele1, allele0, snpCovCompVec,
			      verboseStats, retroData, info);
	lineNum++;
      }
      finImpute2.close();
    }

    ALIGNED_FREE(snpCovCompVec);
    fout.close();
  }

  inline void nullTermMovePos(char *buf, uint64 &pos, bool nullTerm) {
    while (isspace(buf[pos])) pos++;
    while (!isspace(buf[pos])) pos++;
    if (nullTerm) buf[pos++] = '\0';
    else pos++;
  }

  void Bolt::fastStreamImpute2
  (const string &outFile, const vector <string> &impute2Files, const vector <int> &impute2Chroms,
   const string &impute2FidIidFile, double impute2MinMAF, const string &geneticMapFile,
   bool verboseStats, const vector <StatsDataRetroLOCO> &retroData, bool domRecHetTest) const {

    const double impute2CallThresh = 0.95; // seems to have no effect; IMPUTE2 always has total p=1
    FileUtils::AutoGzOfstream fout; fout.openOrExit(outFile);
    
    printStatsHeader(fout, verboseStats, true, retroData);

    MapInterpolater mapInterpolater(geneticMapFile); // ok if no map; then always returns 0
    vector < std::pair <string, string> > impute2IDs = FileUtils::readFidIids(impute2FidIidFile);
    uint Nimpute2 = impute2IDs.size();
    vector <uint64> impute2IndivInds(Nimpute2);
    int numFound = 0;
    for (uint i = 0; i < Nimpute2; i++) {
      impute2IndivInds[i] = snpData.getIndivInd(impute2IDs[i].first, impute2IDs[i].second);
      if (impute2IndivInds[i] != SnpData::IND_MISSING) numFound++;
    }
    cout << endl << "Read " << Nimpute2 << " indivs; using "
	 << numFound << " in filtered PLINK data" << endl;

    FileUtils::AutoGzIfstream finImpute2;

    const uint64 BUF_SIZE = 1ULL<<29; // 0.5 GB
    char *buf = (char *) ALIGNED_MALLOC_UCHARS(BUF_SIZE);

    int threads = omp_get_max_threads();
    //cout << "using " << threads << " threads" << endl;

    for (uint f = 0; f < impute2Files.size(); f++) {
      finImpute2.openOrExit(impute2Files[f]);
      //cout << "file " << impute2Files[f] << endl;
      uint64 bufUsed = 0;
      bool done = false;
      do {
	finImpute2.read(buf+bufUsed, BUF_SIZE-bufUsed);
	bufUsed += finImpute2.gcount();
	//cout << "bufUsed: " << bufUsed << endl;
	done = (bufUsed < BUF_SIZE);
	vector <string> statsStrs(threads); // output lines for SNPs in block
	vector <uint64> nextStarts(threads); // index after last '\n' in block (0 if none)

#pragma omp parallel for
	for (int t = 0; t < threads; t++) {
	  std::ostringstream output;
	  double *snpCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
	  memset(snpCovCompVec+Nstride, 0, Cstride * sizeof(snpCovCompVec[0])); // important!
	  for (uint n = 0; n < Nstride; n++) snpCovCompVec[n] = -9; // initalize dosages to missing
	  double *snpCovCompVecDom = NULL, *snpCovCompVecHet = NULL, *snpCovCompVecRec = NULL;
	  if (domRecHetTest) {
	    snpCovCompVecDom = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
	    snpCovCompVecHet = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
	    snpCovCompVecRec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
	    memset(snpCovCompVecDom+Nstride, 0, Cstride * sizeof(snpCovCompVecDom[0]));
	    memset(snpCovCompVecHet+Nstride, 0, Cstride * sizeof(snpCovCompVecHet[0]));
	    memset(snpCovCompVecRec+Nstride, 0, Cstride * sizeof(snpCovCompVecRec[0]));
	    for (uint n = 0; n < Nstride; n++) snpCovCompVecDom[n] = -9;
	    for (uint n = 0; n < Nstride; n++) snpCovCompVecHet[n] = -9;
	    for (uint n = 0; n < Nstride; n++) snpCovCompVecRec[n] = -9;
	  }
	  char /**snpID,*/ *rsID, *allele1, *allele0;
	  int physpos;

	  uint64 blockStart = bufUsed*t/threads, blockEnd = bufUsed*(t+1)/threads;
	  
	  for (uint64 i = blockEnd-1; i != blockStart-1; i--)
	    if (buf[i] == '\n') {
	      nextStarts[t] = i+1;
	      break;
	    }
	  if (nextStarts[t] == 0) continue; // no SNP line ends in this block
	  uint64 pos = blockStart;
	  while (pos > 0 && buf[pos-1] != '\n') pos--; // find start of first SNP line in block

	  //cout << "pos: " << pos << " nextStarts[" << t << "]: " << nextStarts[t] << endl;
	  
	  while (pos+10 < nextStarts[t]) {
	    //sscanf(buf+pos, "%s%s%d%s%s%n", snpID, rsID, &physpos, allele1, allele0, &deltaPos);
	    //pos += deltaPos;
	    /*snpID = buf+pos;*/ nullTermMovePos(buf, pos, true);
	    rsID = buf+pos; nullTermMovePos(buf, pos, true);
	    physpos = atoi(buf+pos); nullTermMovePos(buf, pos, false);
	    allele1 = buf+pos; nullTermMovePos(buf, pos, true);
	    allele0 = buf+pos; nullTermMovePos(buf, pos, true);

	    if (pos >= nextStarts[t]) break;

	    int chrom = impute2Chroms[f];
	    double genpos = mapInterpolater.interp(chrom, physpos); // 0 if no map
	    double Ncalled = 0; double sum_eij = 0, sum_fij_minus_eij2 = 0; // for INFO
	    // read genotype probabilities
	    for (uint i = 0; i < Nimpute2; i++) {
	      double p11, p10, p00;
	      //sscanf(buf+pos, "%lf%lf%lf%n", &p11, &p10, &p00, &deltaPos);
	      //pos += deltaPos;
	      p11 = atof(buf+pos); nullTermMovePos(buf, pos, false);
	      p10 = atof(buf+pos); nullTermMovePos(buf, pos, false);
	      p00 = atof(buf+pos); nullTermMovePos(buf, pos, false);
	      
	      double pTot = p11 + p10 + p00;
	      double dosage = pTot >= impute2CallThresh ? (2*p11 + p10) / pTot : -9;
	      if (pTot >= impute2CallThresh) { // for INFO
		double eij = dosage;
		double fij = (4*p11 + p10) / pTot;
		sum_eij += eij;
		sum_fij_minus_eij2 += fij - eij*eij;
		Ncalled++;
	      }
	      if (impute2IndivInds[i] != SnpData::IND_MISSING) {
		snpCovCompVec[impute2IndivInds[i]] = dosage;
		if (domRecHetTest) {
		  if (pTot >= impute2CallThresh) {
		    snpCovCompVecDom[impute2IndivInds[i]] = 2 * (p11+p10) / pTot;
		    snpCovCompVecHet[impute2IndivInds[i]] = 2 * p10 / pTot;
		    snpCovCompVecRec[impute2IndivInds[i]] = 2 * p11 / pTot;
		  }
		  else {
		    snpCovCompVecDom[impute2IndivInds[i]] = -9;
		    snpCovCompVecHet[impute2IndivInds[i]] = -9;
		    snpCovCompVecRec[impute2IndivInds[i]] = -9;
		  }
		}
	      }
	    }

	    double thetaHat = sum_eij / (2*Ncalled);
	    double info = thetaHat==0 || thetaHat==1 ? 1 :
	      1 - sum_fij_minus_eij2 / (2*Ncalled*thetaHat*(1-thetaHat));
	    if (std::min(thetaHat, 1-thetaHat) >= impute2MinMAF) {
	      output << getSnpStats(rsID, chrom, physpos, genpos, allele1, allele0, snpCovCompVec,
				    verboseStats, retroData, info);
	      if (domRecHetTest) {
		string testSuffixes[3] = {":Dom", ":Het", ":Rec"};
		string allele1s[3] = {string(allele1)+allele1 + "|" + allele1+allele0,
				      string(allele1)+allele0,
				      string(allele1)+allele1};
		string allele0s[3] = {string(allele0)+allele0,
				      string(allele1)+allele1 + "|" + allele0+allele0,
				      string(allele1)+allele0 + "|" + allele0+allele0};
		double *snpCovCompVecs[3] = {snpCovCompVecDom, snpCovCompVecHet, snpCovCompVecRec};
		for (int ac = 0; ac <= 2; ac++)
		  output << getSnpStats(rsID + testSuffixes[ac], chrom, physpos, genpos,
					allele1s[ac], allele0s[ac], snpCovCompVecs[ac],
					verboseStats, retroData, info);
	      }
	    }
	  }
	  if (domRecHetTest) {
	    ALIGNED_FREE(snpCovCompVecRec);
	    ALIGNED_FREE(snpCovCompVecHet);
	    ALIGNED_FREE(snpCovCompVecDom);
	  }
	  ALIGNED_FREE(snpCovCompVec);
	  statsStrs[t] = output.str();
	}

	for (int t = 0; t < threads; t++)
	  fout << statsStrs[t];

	uint64 maxNextStart = 0;
	for (int t = 0; t < threads; t++)
	  maxNextStart = std::max(maxNextStart, nextStarts[t]);
	bufUsed -= maxNextStart; // copy unparsed chunk (last SNP with partial data in buf)
	//cout << "bufUsed: " << bufUsed << " maxNextStart: " << maxNextStart << endl;
	memmove(buf, buf+maxNextStart, bufUsed);
      } while (!done);

      finImpute2.close();
    }

    ALIGNED_FREE(buf);
    fout.close();
  }

  void Bolt::streamBgen
  (const string &outFile, int f, const string &bgenFile, const string &sampleFile,
   double bgenMinMAF, double bgenMinINFO, const string &geneticMapFile, bool verboseStats,
   const vector <StatsDataRetroLOCO> &retroData, bool domRecHetTest)
    const {

    FileUtils::AutoGzOfstream fout;
    if (f == 0) {
      fout.openOrExit(outFile);
      printStatsHeader(fout, verboseStats, true, retroData);
    }
    else
      fout.openOrExit(outFile, std::ios::app);
    
    MapInterpolater mapInterpolater(geneticMapFile); // ok if no map; then always returns 0
    vector < std::pair <string, string> > sampleIDs = FileUtils::readSampleIDs(sampleFile);
    
    uint Nsample = sampleIDs.size();
    vector <uint64> bgenIndivInds(Nsample);
    int numFound = 0;
    for (uint i = 0; i < Nsample; i++) {
      bgenIndivInds[i] = snpData.getIndivInd(sampleIDs[i].first, sampleIDs[i].second);
      if (bgenIndivInds[i] != SnpData::IND_MISSING) numFound++;
    }
    cout << endl << "Read " << Nsample << " indivs; using "
	 << numFound << " in filtered PLINK data" << endl;

    double *snpCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    memset(snpCovCompVec+Nstride, 0, Cstride * sizeof(snpCovCompVec[0])); // important!
    for (uint n = 0; n < Nstride; n++) snpCovCompVec[n] = -9; // initalize dosages to missing

    FILE *fin = fopen(bgenFile.c_str(), "rb"); assert(fin != NULL);
    uint offset; fread_check(&offset, 4, 1, fin); // cout << "offset: " << offset << endl;
    uint H; fread_check(&H, 4, 1, fin); // cout << "H: " << H << endl;
    uint Mbgen; fread_check(&Mbgen, 4, 1, fin); cout << "BGEN snpBlocks (Mbgen): " << Mbgen << endl;
    uint Nbgen; fread_check(&Nbgen, 4, 1, fin); cout << "BGEN samples (Nbgen): " << Nbgen << endl;
    if (Nbgen != Nsample) {
      cerr << "ERROR: Number of samples in BGEN header does not match sample file" << endl;
      exit(1);
    }
    fseek_check(fin, offset+4, SEEK_SET);
    char snpID[65536], rsID[65536], chrStr[65536];
    char *allele1, *allele0;
    uint maxLA = 65536, maxLB = 65536;
    allele1 = (char *) malloc(maxLA+1);
    allele0 = (char *) malloc(maxLB+1);
    uchar *zBuf = ALIGNED_MALLOC_UCHARS(6*Nbgen);
    ushort *shortBuf = (ushort *) ALIGNED_MALLOC_UCHARS(6*Nbgen);
  
    for (uint mbgen = 0; mbgen < Mbgen; mbgen++) {
      uint Nrow; fread_check(&Nrow, 4, 1, fin); // cout << "Nrow: " << Nrow << " " << std::flush;
      if (Nrow != Nbgen) {
	cerr << "ERROR: Nrow = " << Nrow << " does not match Nbgen = " << Nbgen << endl;
	exit(1);
      }
      ushort LS; fread_check(&LS, 2, 1, fin); // cout << "LS: " << LS << " " << std::flush;
      fread_check(snpID, 1, LS, fin); snpID[LS] = '\0'; // cout << "snpID: " << string(snpID) << " " << std::flush;
      ushort LR; fread_check(&LR, 2, 1, fin); // cout << "LR: " << LR << " " << std::flush;
      fread_check(rsID, 1, LR, fin); rsID[LR] = '\0'; // cout << "rsID: " << string(rsID) << " " << std::flush;
      ushort LC; fread_check(&LC, 2, 1, fin); // cout << "LC: " << LC << " " << std::flush;
      fread_check(chrStr, 1, LC, fin); chrStr[LC] = '\0';
      int chrom = SnpData::chrStrToInt(chrStr, Nautosomes);
      if (chrom == -1) {
	cerr << "ERROR: Invalid chrom (expecting integer 1-" << Nautosomes+1
	     << " or X,XY,PAR1,PAR2): " << string(chrStr) << endl;
	exit(1);
      }
      uint physpos; fread_check(&physpos, 4, 1, fin); // cout << "physpos: " << physpos << " " << std::flush;
      double genpos = mapInterpolater.interp(chrom, physpos); // 0 if no map
      uint LA; fread_check(&LA, 4, 1, fin); // cout << "LA: " << LA << " " << std::flush;
      if (LA > maxLA) {
	maxLA = 2*LA;
	free(allele1);
	allele1 = (char *) malloc(maxLA+1);
      }
      fread_check(allele1, 1, LA, fin); allele1[LA] = '\0';
      uint LB; fread_check(&LB, 4, 1, fin); // cout << "LB: " << LB << " " << std::flush;
      if (LB > maxLB) {
	maxLB = 2*LB;
	free(allele0);
	allele0 = (char *) malloc(maxLB+1);
      }
      fread_check(allele0, 1, LB, fin); allele0[LB] = '\0';
      uint zLen; fread_check(&zLen, 4, 1, fin); // cout << "zLen: " << zLen << endl;
      fread_check(zBuf, 1, zLen, fin);
      uLongf destLen = 6*Nbgen;
      if (uncompress((Bytef *) shortBuf, &destLen, zBuf, zLen) != Z_OK || destLen != 6*Nbgen) {
	cerr << "ERROR: uncompress() failed" << endl;
	exit(1);
      }

      // read genotype probabilities
      double sum_eij = 0, sum_fij_minus_eij2 = 0; // for INFO
      const double scale = 1.0/32768;
      for (uint i = 0; i < Nbgen; i++) {
	double p11 = shortBuf[3*i] * scale;
	double p10 = shortBuf[3*i+1] * scale;
	double p00 = shortBuf[3*i+2] * scale;

	double pTot = p11 + p10 + p00;
	double dosage = (2*p11 + p10) / pTot;
	double eij = dosage;
	double fij = (4*p11 + p10) / pTot;
	sum_eij += eij;
	sum_fij_minus_eij2 += fij - eij*eij;

	if (bgenIndivInds[i] != SnpData::IND_MISSING)
	  snpCovCompVec[bgenIndivInds[i]] = dosage;
      }

      double thetaHat = sum_eij / (2*Nbgen);
      double info = thetaHat==0 || thetaHat==1 ? 1 :
	1 - sum_fij_minus_eij2 / (2*Nbgen*thetaHat*(1-thetaHat));
      if (std::min(thetaHat, 1-thetaHat) >= bgenMinMAF && info >= bgenMinINFO) {
	string snpName = string(rsID)=="." ? snpID : rsID;
	fout << getSnpStats(snpName, chrom, physpos, genpos, allele1, allele0, snpCovCompVec,
			    verboseStats, retroData, info);
	if (domRecHetTest) {
	  /*
	  uchar *alleleCounts = ALIGNED_MALLOC_UCHARS(Nbgen);
	  for (uint i = 0; i < Nbgen; i++) {
	    if (shortBuf[3*i] >= shortBuf[3*i+1] && shortBuf[3*i] >= shortBuf[3*i+2])
	      alleleCounts[i] = 2;
	    else if (shortBuf[3*i+1] >= shortBuf[3*i+2])
	      alleleCounts[i] = 1;
	    else
	      alleleCounts[i] = 0;
	  }
	  */
	  string testSuffixes[3] = {":Dom", ":Het", ":Rec"};
	  string allele1s[3] = {string(allele1)+allele1 + "|" + allele1+allele0,
				string(allele1)+allele0,
				string(allele1)+allele1};
	  string allele0s[3] = {string(allele0)+allele0,
				string(allele1)+allele1 + "|" + allele0+allele0,
				string(allele1)+allele0 + "|" + allele0+allele0};
	  for (int ac = 0; ac <= 2; ac++) {
	    for (uint i = 0; i < Nbgen; i++)
	      if (bgenIndivInds[i] != SnpData::IND_MISSING) {
		double p11 = shortBuf[3*i] * scale;
		double p10 = shortBuf[3*i+1] * scale;
		double p00 = shortBuf[3*i+2] * scale;
		double pTot = p11 + p10 + p00;
		if (ac == 0)
		  snpCovCompVec[bgenIndivInds[i]] = //2 * (alleleCounts[i] != ac);
		    2 * (p11+p10) / pTot;
		else if (ac == 1)
		  snpCovCompVec[bgenIndivInds[i]] = //2 * (alleleCounts[i] == ac);
		    2 * p10 / pTot;
		else if (ac == 2)
		  snpCovCompVec[bgenIndivInds[i]] = //2 * (alleleCounts[i] == ac);
		    2 * p11 / pTot;
	      }
	    fout << getSnpStats(snpName + testSuffixes[ac], chrom, physpos, genpos, allele1s[ac],
				allele0s[ac], snpCovCompVec, verboseStats, retroData, info);
	  }
	  //ALIGNED_FREE(alleleCounts);
	}
      }
    }

    free(allele0);
    free(allele1);

    ALIGNED_FREE(shortBuf);
    ALIGNED_FREE(zBuf);
    ALIGNED_FREE(snpCovCompVec);

    fclose(fin);
    fout.close();
  }

  string Bolt::getSnpStatsBgen2(uchar *buf, uint bufLen, const uchar *zBuf, uint zBufLen,
				uint Nbgen, const vector <uint64> &bgenIndivInds,
				const string &snpName, int chrom, int physpos,double genpos,
				const string &allele1, const string &allele0,
				double snpCovCompVec[], bool verboseStats,
				const vector <StatsDataRetroLOCO> &retroData, bool domRecHetTest,
				double bgenMinMAF, double bgenMinINFO) const {

    /********** decompress and check genotype probability block **********/

    //cout << "bufLen = " << bufLen << " zBufLen = " << zBufLen << endl;
    uLongf destLen = bufLen;
    if (uncompress(buf, &destLen, zBuf, zBufLen) != Z_OK || destLen != bufLen) {
      cerr << "ERROR: uncompress() failed" << endl;
      exit(1);
    }
    uchar *bufAt = buf;
    uint N = bufAt[0]|(bufAt[1]<<8)|(bufAt[2]<<16)|(bufAt[3]<<24); bufAt += 4;
    if (N != Nbgen) {
      cerr << "ERROR: " << snpName << " has N = " << N << " (mismatch with header block)" << endl;
      exit(1);
    }
    uint K = bufAt[0]|(bufAt[1]<<8); bufAt += 2;
    if (K != 2U) {
      cerr << "ERROR: " << snpName << " has K = " << K << " (non-bi-allelic)" << endl;
      exit(1);
    }
    uint Pmin = *bufAt; bufAt++;
    if (Pmin != 2U) {
      cerr << "ERROR: " << snpName << " has minimum ploidy = " << Pmin << " (not 2)" << endl;
      exit(1);
    }
    uint Pmax = *bufAt; bufAt++;
    if (Pmax != 2U) {
      cerr << "ERROR: " << snpName << " has maximum ploidy = " << Pmax << " (not 2)" << endl;
      exit(1);
    }
    const uchar *ploidyMissBytes = bufAt;
    for (uint i = 0; i < N; i++) {
      uint ploidyMiss = *bufAt; bufAt++;
      if (ploidyMiss != 2U && ploidyMiss != 130U) {
	cerr << "ERROR: " << snpName << " has ploidy/missingness byte = " << ploidyMiss
	     << " (not 2 or 130)" << endl;
	exit(1);
      }
    }
    uint Phased = *bufAt; bufAt++;
    if (Phased != 0U && Phased != 1U) {
      cerr << "ERROR: " << snpName << " has Phased = " << Phased << " (not 0 or 1)" << endl;
      exit(1);
    }
    uint B = *bufAt; bufAt++;
    if (B != 8U) {
      cerr << "ERROR: " << snpName << " has B = " << B << " (not 8)" << endl;
      exit(1);
    }

    /********** compute MAF and INFO; apply filtering thresholds **********/

    double lut[256];
    for (int i = 0; i <= 255; i++)
      lut[i] = i/255.0;

    int Nnonmiss = 0;
    double sum_eij = 0, sum_fij_minus_eij2 = 0; // for INFO
    for (uint i = 0; i < N; i++) {
      if (ploidyMissBytes[i] == 130U) { bufAt += 2; continue; }
      Nnonmiss++;
      double p11 = lut[*bufAt]; bufAt++; // note: if phased, will contain p(hap1)==1
      double p10 = lut[*bufAt]; bufAt++; // note: if phased, will contain p(hap2)==1
      double dosage = (Phased==0U ? 2 : 1) * p11 + p10;
      double eij = dosage;
      double fij = (Phased==0U ? 4*p11 + p10 : p11 + 2*p11*p10 + p10);
      sum_eij += eij;
      sum_fij_minus_eij2 += fij - eij*eij;
    }
    double thetaHat = sum_eij / (2*Nnonmiss);
    double info = thetaHat==0 || thetaHat==1 ? 1 :
      1 - sum_fij_minus_eij2 / (2*Nnonmiss*thetaHat*(1-thetaHat));
    double maf = std::min(thetaHat, 1-thetaHat);

    if (maf < bgenMinMAF || info < bgenMinINFO || Nnonmiss==0) return "";

    /********** compute association statistics **********/

    std::ostringstream oss;

    // reread dosages and copy to buffer in correct order
    bufAt -= 2*N; // rewind
    for (uint i = 0; i < N; i++) {
      double p11 = lut[*bufAt]; bufAt++;
      double p10 = lut[*bufAt]; bufAt++;
      double dosage = (Phased==0U ? 2 : 1) * p11 + p10;
      if (bgenIndivInds[i] != SnpData::IND_MISSING)
	snpCovCompVec[bgenIndivInds[i]] = ploidyMissBytes[i]==2U ? dosage : 2*thetaHat;
    }

    oss << getSnpStats(snpName, chrom, physpos, genpos, allele1, allele0, snpCovCompVec,
		       verboseStats, retroData, info);
    if (domRecHetTest) {
      string testSuffixes[3] = {":Dom", ":Het", ":Rec"};
      string allele1s[3] = {string(allele1)+allele1 + "|" + allele1+allele0,
			    string(allele1)+allele0,
			    string(allele1)+allele1};
      string allele0s[3] = {string(allele0)+allele0,
			    string(allele1)+allele1 + "|" + allele0+allele0,
			    string(allele1)+allele0 + "|" + allele0+allele0};
      for (int ac = 0; ac <= 2; ac++) {
	bufAt -= 2*N; // rewind
	double sum = 0;
	for (uint i = 0; i < N; i++) {
	  double p11 = lut[*bufAt]; bufAt++;
	  double p10 = lut[*bufAt]; bufAt++;
	  if (ploidyMissBytes[i]==2U && bgenIndivInds[i] != SnpData::IND_MISSING) {
	    if (ac == 0)
	      snpCovCompVec[bgenIndivInds[i]] = //2 * (alleleCounts[i] != ac);
		2 * (p11+p10);
	    else if (ac == 1)
	      snpCovCompVec[bgenIndivInds[i]] = //2 * (alleleCounts[i] == ac);
		2 * p10;
	    else if (ac == 2)
	      snpCovCompVec[bgenIndivInds[i]] = //2 * (alleleCounts[i] == ac);
		2 * p11;
	    sum += snpCovCompVec[bgenIndivInds[i]];
	  }
	}
	double mean = sum / Nnonmiss;
	for (uint i = 0; i < N; i++)
	  if (ploidyMissBytes[i]==130U && bgenIndivInds[i] != SnpData::IND_MISSING)
	    snpCovCompVec[bgenIndivInds[i]] = mean;

	oss << getSnpStats(snpName + testSuffixes[ac], chrom, physpos, genpos, allele1s[ac],
			   allele0s[ac], snpCovCompVec, verboseStats, retroData, info);
      }
    }
    
    return oss.str();
  }

  void Bolt::streamBgen2
  (const string &outFile, int f, const string &bgenFile, const string &sampleFile,
   double bgenMinMAF, double bgenMinINFO, const string &geneticMapFile, bool verboseStats,
   const vector <StatsDataRetroLOCO> &retroData, bool domRecHetTest, int threads)
    const {

#ifdef USE_MKL
    mkl_set_num_threads(1); // don't use nested threading in DGEMV calls (for covariate projection)
#endif

    FileUtils::AutoGzOfstream fout;
    if (f == 0) {
      fout.openOrExit(outFile);
      printStatsHeader(fout, verboseStats, true, retroData);
    }
    else
      fout.openOrExit(outFile, std::ios::app);
    
    MapInterpolater mapInterpolater(geneticMapFile); // ok if no map; then always returns 0
    vector < std::pair <string, string> > sampleIDs = FileUtils::readSampleIDs(sampleFile);
    
    uint Nsample = sampleIDs.size();
    vector <uint64> bgenIndivInds(Nsample);
    int numFound = 0;
    for (uint i = 0; i < Nsample; i++) {
      bgenIndivInds[i] = snpData.getIndivInd(sampleIDs[i].first, sampleIDs[i].second);
      if (bgenIndivInds[i] != SnpData::IND_MISSING) numFound++;
    }
    cout << endl << "Read " << Nsample << " indivs; using "
	 << numFound << " in filtered PLINK data" << endl;

    // allocate thread-specific memory buffers
    double *snpCovCompVecs[threads];
    for (int t = 0; t < threads; t++) {
      snpCovCompVecs[t] = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
      memset(snpCovCompVecs[t]+Nstride, 0, Cstride * sizeof(snpCovCompVecs[t][0])); // important!
      for (uint n = 0; n < Nstride; n++) snpCovCompVecs[t][n] = -9; // initalize dosages to missing
    }
    vector < vector <uchar> > bufs(threads);

    /********** READ HEADER **********/

    FILE *fin = fopen(bgenFile.c_str(), "rb"); assert(fin != NULL);
    uint offset; fread_check(&offset, 4, 1, fin); //cout << "offset: " << offset << endl;
    uint L_H; fread_check(&L_H, 4, 1, fin); //cout << "L_H: " << L_H << endl;
    uint Mbgen; fread_check(&Mbgen, 4, 1, fin); cout << "snpBlocks (Mbgen): " << Mbgen << endl;
    assert(Mbgen != 0);
    uint Nbgen; fread_check(&Nbgen, 4, 1, fin); cout << "samples (Nbgen): " << Nbgen << endl;
    if (Nbgen != Nsample) {
      cerr << "ERROR: Number of samples in BGEN header does not match sample file" << endl;
      exit(1);
    }
    char magic[5]; fread_check(magic, 1, 4, fin); magic[4] = '\0'; //cout << "magic bytes: " << string(magic) << endl;
    fseek_check(fin, L_H-20, SEEK_CUR); //cout << "skipping L_H-20 = " << L_H-20 << " bytes (free data area)" << endl;
    uint flags; fread_check(&flags, 4, 1, fin); //cout << "flags: " << flags << endl;
    uint CompressedSNPBlocks = flags&3; cout << "CompressedSNPBlocks: " << CompressedSNPBlocks << endl;
    assert(CompressedSNPBlocks==1); // REQUIRE CompressedSNPBlocks==1
    uint Layout = (flags>>2)&0xf; cout << "Layout: " << Layout << endl;
    assert(Layout==1 || Layout==2); // REQUIRE Layout==1 or Layout==2

    //uint SampleIdentifiers = flags>>31; //cout << "SampleIdentifiers: " << SampleIdentifiers << endl;
    fseek_check(fin, offset+4, SEEK_SET);

    /********** READ SNP BLOCKS IN BATCHES **********/

    const int B_MAX = 400; // number of SNPs to process in one batch (for multi-threading)

    char snpID[65536], rsID[65536], chrStr[65536];
    char *allele1, *allele0;
    uint maxLA = 65536, maxLB = 65536;
    allele1 = (char *) malloc(maxLA+1);
    allele0 = (char *) malloc(maxLB+1);

    // during single-threaded reading of block, store SNP data for later multi-threaded processing
    vector <string> snpNames(B_MAX);
    vector <int> chroms(B_MAX);
    vector <int> bps(B_MAX);
    vector <double> gps(B_MAX);
    vector <string> allele1s(B_MAX), allele0s(B_MAX);
    vector <string> outStrs(B_MAX);
    vector < vector <uchar> > zBufs(B_MAX);
    vector <uint> zBufLens(B_MAX), bufLens(B_MAX);
    
    Timer timer;
    int B = 0; // current block size
    for (uint mbgen = 0; mbgen < Mbgen; mbgen++) {
      ushort LS; fread_check(&LS, 2, 1, fin); // cout << "LS: " << LS << " " << std::flush;
      fread_check(snpID, 1, LS, fin); snpID[LS] = '\0'; // cout << "snpID: " << string(snpID) << " " << std::flush;
      ushort LR; fread_check(&LR, 2, 1, fin); // cout << "LR: " << LR << " " << std::flush;
      fread_check(rsID, 1, LR, fin); rsID[LR] = '\0'; // cout << "rsID: " << string(rsID) << " " << std::flush;
      snpNames[B] = string(rsID)=="." ? snpID : rsID;

      ushort LC; fread_check(&LC, 2, 1, fin); // cout << "LC: " << LC << " " << std::flush;
      fread_check(chrStr, 1, LC, fin); chrStr[LC] = '\0';
      int chrom = SnpData::chrStrToInt(chrStr, Nautosomes);
      if (chrom == -1) {
	cerr << "ERROR: Invalid chrom (expecting integer 1-" << Nautosomes+1
	     << " or X,XY,PAR1,PAR2): " << string(chrStr) << endl;
	exit(1);
      }
      chroms[B] = chrom;

      uint physpos; fread_check(&physpos, 4, 1, fin); // cout << "physpos: " << physpos << " " << std::flush;
      bps[B] = physpos;

      double genpos = mapInterpolater.interp(chrom, physpos); // 0 if no map
      gps[B] = genpos;

      ushort K; fread_check(&K, 2, 1, fin); //cout << "K: " << K << endl;
      if (K != 2) {
	cerr << "ERROR: Non-bi-allelic variant found: " << K << " alleles" << endl;
	exit(1);
      }

      uint LA; fread_check(&LA, 4, 1, fin); // cout << "LA: " << LA << " " << std::flush;
      if (LA > maxLA) {
	maxLA = 2*LA;
	free(allele1);
	allele1 = (char *) malloc(maxLA+1);
      }
      fread_check(allele1, 1, LA, fin); allele1[LA] = '\0';
      allele1s[B] = string(allele1);
      uint LB; fread_check(&LB, 4, 1, fin); // cout << "LB: " << LB << " " << std::flush;
      if (LB > maxLB) {
	maxLB = 2*LB;
	free(allele0);
	allele0 = (char *) malloc(maxLB+1);
      }
      fread_check(allele0, 1, LB, fin); allele0[LB] = '\0';
      allele0s[B] = string(allele0);

      uint C; fread_check(&C, 4, 1, fin); //cout << "C: " << C << endl;
      if (C > zBufs[B].size()) zBufs[B].resize(C-4);
      uint D; fread_check(&D, 4, 1, fin); //cout << "D: " << D << endl;
      zBufLens[B] = C-4; bufLens[B] = D;
      fread_check(&zBufs[B][0], 1, C-4, fin);

      B++;
      if (B == B_MAX || mbgen+1 == Mbgen) { // process the block of SNPs using multi-threading
#ifdef USE_MKL
#pragma omp parallel for schedule(dynamic)
#endif
	for (int b = 0; b < B; b++) {
	  int t = omp_get_thread_num();
	  if (bufLens[b] > bufs[t].size()) bufs[t].resize(bufLens[b]);
	  outStrs[b] = getSnpStatsBgen2(&bufs[t][0], bufLens[b], &zBufs[b][0], zBufLens[b], Nbgen,
					bgenIndivInds, snpNames[b], chroms[b], bps[b], gps[b],
					allele1s[b], allele0s[b], snpCovCompVecs[t], verboseStats,
					retroData, domRecHetTest, bgenMinMAF, bgenMinINFO);
	}

	for (int b = 0; b < B; b++)
	  fout << outStrs[b];

	B = 0; // reset current block size
      }
      if (mbgen % 100000 == 99999)
	cout << "At SNP " << mbgen+1 << "; time for block: " << timer.update_time() << endl;
    }

    free(allele0);
    free(allele1);

    for (int t = 0; t < threads; t++)
      ALIGNED_FREE(snpCovCompVecs[t]);

    fclose(fin);
    fout.close();

#ifdef USE_MKL
    mkl_set_num_threads(threads); // reset MKL threading
#endif
  }
}
