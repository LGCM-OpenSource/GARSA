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

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "Types.hpp"
#include "Timer.hpp"
#include "SnpData.hpp"
#include "CovariateBasis.hpp"
#include "NumericUtils.hpp"
#include "LapackConst.hpp"
#include "MemoryUtils.hpp"
#include "Jackknife.hpp"
#include "StatsUtils.hpp"
#include "MatrixUtils.hpp"
#include "NonlinearOptMulti.hpp"
#include "Bolt.hpp"

namespace LMM {

  namespace ublas = boost::numeric::ublas;
  using std::vector;
  using std::string;
  using std::pair;
  using std::cout;
  using std::cerr;
  using std::endl;

  /**
   * for k=1..VCs, computes (Theta_k[b] - coeffI*I) * x[b] = (Z_k[b] * Z_k[b]' - coeffI*I) * x[b],
   * where
   *     Z_k[b] = sqrt(vcXscale2s[k]) X_k
   *     Theta_k[b] = vcXscale2s[k] X_k * X_k'
   * (see conjGradSolveW for more info)
   *
   * multCovCompVecs: (out) VCs x B x (Nstride+Cstride)
   * xCovCompVecs: (in) B x (Nstride+Cstride)
   * h2Xscale2batches: (in) (1+VCs) x B table of combined scale factors for easy application
   */
  void Bolt::multThetaMinusIs(double multCovCompVecs[], const double xCovCompVecs[],
			      const uchar snpVCnums[], const vector <double> &vcXscale2s,
			      uint64 B, double coeffI) const {
#ifdef VERBOSE
    Timer timer;
    printf("  Multiplying solutions by variance components... ");
    fflush(stdout);
#endif
    int VCs = vcXscale2s.size()-1;
    const uint64 BxNC = B*(Nstride+Cstride);
    memset(multCovCompVecs, 0, VCs * BxNC * sizeof(multCovCompVecs[0])); // initialize answers to 0
    
    double *snpCovCompVecBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * (Nstride+Cstride));
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(omp_get_max_threads() * 256*sizeof(*work));
    double *XtransVecsBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * B);

    // for each SNP block...
    for (uint64 m0 = 0; m0 < M; m0 += mBlockMultX) {

      // (1) load the SNPs in parallel into Xblock [Nstride+Cstride x block size]
      uint64 mBlockMultXCrop = std::min(M, m0+mBlockMultX) - m0;
#pragma omp parallel for
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (projMaskSnps[m] && snpVCnums[m]) // build snp vector + sign-flipped covar comps
	  buildMaskedSnpNegCovCompVec(snpCovCompVecBlock + mPlus * (Nstride+Cstride), m,
				      work + (omp_get_thread_num()<<8));
	else
	  memset(snpCovCompVecBlock + mPlus * (Nstride+Cstride), 0,
		 (Nstride+Cstride) * sizeof(snpCovCompVecBlock[0]));
      }

      // (2) multiply Xblock' [block size x Nstride+Cstride] * xVecs [Nstride+Cstride x B]
      //              = XtransVecsBlock [block size x B]
      //     (note that Xblock' has NEG CCVecs while xVecs has POS CCVecs)
      {
	char TRANSA_ = 'T';
	char TRANSB_ = 'N';
	int M_ = B;
	int N_ = mBlockMultXCrop;
	int K_ = Nstride+Cstride;
	double ALPHA_ = 1.0;
	const double *A_ = xCovCompVecs;
	int LDA_ = Nstride+Cstride;
	double *B_ = snpCovCompVecBlock;
	int LDB_ = Nstride+Cstride;
	double BETA_ = 0.0;
	double *C_ = XtransVecsBlock;
	int LDC_ = B;

	DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		    &BETA_, C_, &LDC_);
      }
      
      // (4) for each snp individually in turn:
      //     multiply Xblock_mPlus [NCstride x 1] * XtransVecsBlock_mPlus [1 x B]
      //              = snp's contrib. to (k-1)^th result block (X_k X_k') * xVecs [NCstride x B]
      //     directly accumulate results in (k-1)^th answer mult [NCstride x B]
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (projMaskSnps[m] && snpVCnums[m]) {
	  double *snpCovCompVec = snpCovCompVecBlock + mPlus * (Nstride+Cstride);
	  // note that Xblock was loaded with NEG CCVecs; now we need to sign-flip CCVecs
	  for (uint64 n = Nstride; n < Nstride+Cstride; n++)
	    snpCovCompVec[n] *= -1;

	  // update result block for snpVCnums[m]: add XtransVecsBlock_mPlus * snpCovCompVec (DGER)
	  {
	    int M_ = Nstride+Cstride;
	    int N_ = B;
	    double ALPHA_ = 1;
	    double *X_ = snpCovCompVec;
	    int INCX_ = 1;
	    double *Y_ = XtransVecsBlock + mPlus * B;
	    int INCY_ = 1;
	    double *A_ = multCovCompVecs + (snpVCnums[m]-1) * BxNC;
	    int LDA_ = Nstride+Cstride;
	    DGER_MACRO(&M_, &N_, &ALPHA_, X_, &INCX_, Y_, &INCY_, A_, &LDA_);
	  }
	}
      }
    }
    ALIGNED_FREE(XtransVecsBlock);
    ALIGNED_FREE(work);
    ALIGNED_FREE(snpCovCompVecBlock);

    // multiply vcXscale2s and subtract identity contributions
    for (int v = 0; v < VCs; v++)
      for (uint64 bnc = 0; bnc < BxNC; bnc++)
	multCovCompVecs[v*BxNC + bnc] =
	  vcXscale2s[v+1] * multCovCompVecs[v*BxNC + bnc] - coeffI * xCovCompVecs[bnc];
#ifdef VERBOSE
    printf("time=%.2f\n", timer.update_time());
    fflush(stdout);
#endif
  }

  void rightMultiT(double *x, uint64 D, uint64 stride, const ublas::matrix <double> &mat) {
    vector <double> tmpD(D);
    for (uint64 d = 0; d < D; d++)
      for (uint64 i = 0; i < D; i++)
	tmpD[d] += x[i*stride] * mat(d, i);
    for (uint64 d = 0; d < D; d++)
      x[d*stride] = tmpD[d];
  }

  double Bolt::dotMultiCovCompVecs(const double xCovCompVecs[], const double yCovCompVecs[],
				   uint64 D) const {
    double ret = 0;
    for (uint64 d = 0; d < D; d++)
      ret += dotCovCompVecs(xCovCompVecs+d*(Nstride+Cstride), yCovCompVecs+d*(Nstride+Cstride));
    return ret;
  }

  void Bolt::computeMultiProjNorm2s(double projNorm2s[], const double xCovCompVecs[], uint64 D,
				    uint64 B) const {
    for (uint64 b = 0; b < B; b++) {
      projNorm2s[b] = 0;
      for (uint64 d = 0; d < D; d++)
	projNorm2s[b] += computeProjNorm2(xCovCompVecs + (b*D+d)*(Nstride+Cstride));
    }
  }

  /**
   * TODO
   * NOTE: VinvyMultiCovCompVecs is destroyed if Vegs[0] is non-identity!
   */
  ublas::vector <double> Bolt::updateVegs(vector < ublas::matrix <double> > &Vegs,
					  double VinvyMultiCovCompVecs[], const uchar snpVCnums[],
					  const vector <double> &vcXscale2s, int MCtrials) const {

    int VCs = Vegs.size()-1;
    uint64 D = Vegs[0].size1();
    vector < ublas::matrix <double> > deltaVegs(1+VCs, ublas::zero_matrix <double> (D, D)),
      VegXscales(1+VCs);
    for (int k = 0; k <= VCs; k++)
      VegXscales[k] = (k == 0 ? 1 : sqrt(vcXscale2s[k])) * Vegs[k];
    vector <double> denoms(1+VCs);
    denoms[0] = (double) (Nused-Cindep);
    for (uint64 m = 0; m < M; m++)
      if (projMaskSnps[m] && snpVCnums[m])
	denoms[snpVCnums[m]]++;

    uint64 DxNC = D * (Nstride+Cstride);

    uint64 B = MCtrials+1;
    double *snpCovCompVecBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * (Nstride+Cstride));
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(omp_get_max_threads() * 256*sizeof(*work));
    double *XtransVecsBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * B * D);

    // for each SNP block...
    for (uint64 m0 = 0; m0 < M; m0 += mBlockMultX) {

      // (1) load the SNPs in parallel into Xblock [Nstride+Cstride x block size]
      uint64 mBlockMultXCrop = std::min(M, m0+mBlockMultX) - m0;
#pragma omp parallel for
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (projMaskSnps[m] && snpVCnums[m]) // build snp vector + sign-flipped covar comps
	  buildMaskedSnpNegCovCompVec(snpCovCompVecBlock + mPlus * (Nstride+Cstride), m,
				      work + (omp_get_thread_num()<<8));
	else
	  memset(snpCovCompVecBlock + mPlus * (Nstride+Cstride), 0,
		 (Nstride+Cstride) * sizeof(snpCovCompVecBlock[0]));
      }

      // (2) multiply Xblock' [block size x Nstride+Cstride] * xMultiVecs [Nstride+Cstride x BxD]
      //              = XtransVecsBlock [block size x BxD]
      //     (note that Xblock' has NEG CCVecs while xVecs has POS CCVecs)
      {
#ifdef MEASURE_DGEMM
      //Timer timer;
      unsigned long long tsc = Timer::rdtsc();
#endif
	char TRANSA_ = 'T';
	char TRANSB_ = 'N';
	int M_ = B * D;
	int N_ = mBlockMultXCrop;
	int K_ = Nstride+Cstride;
	double ALPHA_ = 1.0;
	const double *A_ = VinvyMultiCovCompVecs;
	int LDA_ = Nstride+Cstride;
	double *B_ = snpCovCompVecBlock;
	int LDB_ = Nstride+Cstride;
	double BETA_ = 0.0;
	double *C_ = XtransVecsBlock;
	int LDC_ = B * D;

	DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		    &BETA_, C_, &LDC_);
#ifdef MEASURE_DGEMM
      dgemmTicks += Timer::rdtsc() - tsc;
      //dgemmTicks += timer.update_time();
#endif
      }
      
      // (3) SNP-wise multiply XtransVecsBlock by appropriate Veg factor along with sqrt(Xscale2s)
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (snpVCnums[m]) { // SNP belongs to a variance component
	  uint64 k = snpVCnums[m];
	  for (uint64 b = 0; b < B; b++) {
	    uint64 coeffStart = mPlus*B*D + b*D;
	    rightMultiT(XtransVecsBlock + coeffStart, D, 1, VegXscales[k]);
	    double mult = (int) b == MCtrials ? 1 : -1.0/MCtrials;
	    for (uint64 di = 0; di < D; di++)
	      for (uint64 dj = 0; dj <= di; dj++)
		deltaVegs[k](di, dj) += mult * XtransVecsBlock[coeffStart+di] *
		  XtransVecsBlock[coeffStart+dj];
	  }
	}
      }
      
    }
    ALIGNED_FREE(XtransVecsBlock);
    ALIGNED_FREE(work);
    ALIGNED_FREE(snpCovCompVecBlock);

    // compute env variance parameter matrix Vegs[0] updates
    // directly use ***and destroy*** VinvyMultiCovCompVecs: need to rightMultiT by Vegs[0]
    for (int t = 0; t <= MCtrials; t++) {
      for (uint64 nc = 0; nc < Nstride+Cstride; nc++) // replace Vinvy with Ue-hats (env)
	rightMultiT(VinvyMultiCovCompVecs + t*DxNC + nc, D, Nstride+Cstride, Vegs[0]);
      double mult = t == MCtrials ? 1 : -1.0/MCtrials;      
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++)
	  deltaVegs[0](di, dj) += mult *
	    dotCovCompVecs(VinvyMultiCovCompVecs + t*DxNC + di*(Nstride+Cstride),
			   VinvyMultiCovCompVecs + t*DxNC + dj*(Nstride+Cstride));
    }
    
    for (int k = 0; k <= VCs; k++) {
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++)
	  deltaVegs[k](dj, di) = deltaVegs[k](di, dj); // symmetrize
      Vegs[k] += 1/denoms[k] * deltaVegs[k];
    }

    int numPars = (1+VCs) * D*(D+1)/2;
    ublas::vector <double> grad(numPars);
    int curPar = 0;
    for (int k = 0; k <= VCs; k++)
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++)
	  grad(curPar++) = (di == dj ? 0.5 : 1.0) * deltaVegs[k](di, dj);
    return grad;
  }

  ublas::vector <double> Bolt::computeMCgrad(const double VinvyMultiCovCompVecs[], uint64 D,
					     const uchar snpVCnums[], int VCs,
					     const vector <double> &vcXscale2s, int MCtrials)
    const {

    int numPars = (1+VCs) * D*(D+1)/2;
    ublas::vector <double> grad = ublas::zero_vector <double> (numPars);
    uint64 DxNC = D * (Nstride+Cstride);
    uint64 B = MCtrials+1;
    double *snpCovCompVecBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * (Nstride+Cstride));
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(omp_get_max_threads() * 256*sizeof(*work));
    double *XtransVecsBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * B * D);

    // for each SNP block...
    for (uint64 m0 = 0; m0 < M; m0 += mBlockMultX) {

      // (1) load the SNPs in parallel into Xblock [Nstride+Cstride x block size]
      uint64 mBlockMultXCrop = std::min(M, m0+mBlockMultX) - m0;
#pragma omp parallel for
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (projMaskSnps[m] && snpVCnums[m]) // build snp vector + sign-flipped covar comps
	  buildMaskedSnpNegCovCompVec(snpCovCompVecBlock + mPlus * (Nstride+Cstride), m,
				      work + (omp_get_thread_num()<<8));
	else
	  memset(snpCovCompVecBlock + mPlus * (Nstride+Cstride), 0,
		 (Nstride+Cstride) * sizeof(snpCovCompVecBlock[0]));
      }

      // (2) multiply Xblock' [block size x Nstride+Cstride] * xMultiVecs [Nstride+Cstride x BxD]
      //              = XtransVecsBlock [block size x BxD]
      //     (note that Xblock' has NEG CCVecs while xVecs has POS CCVecs)
      {
#ifdef MEASURE_DGEMM
      //Timer timer;
      unsigned long long tsc = Timer::rdtsc();
#endif
	char TRANSA_ = 'T';
	char TRANSB_ = 'N';
	int M_ = B * D;
	int N_ = mBlockMultXCrop;
	int K_ = Nstride+Cstride;
	double ALPHA_ = 1.0;
	const double *A_ = VinvyMultiCovCompVecs;
	int LDA_ = Nstride+Cstride;
	double *B_ = snpCovCompVecBlock;
	int LDB_ = Nstride+Cstride;
	double BETA_ = 0.0;
	double *C_ = XtransVecsBlock;
	int LDC_ = B * D;

	DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		    &BETA_, C_, &LDC_);
#ifdef MEASURE_DGEMM
      dgemmTicks += Timer::rdtsc() - tsc;
      //dgemmTicks += timer.update_time();
#endif
      }
      
      // (3) SNP-wise multiply XtransVecsBlock by appropriate Veg factor along with sqrt(Xscale2s)
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (snpVCnums[m]) { // SNP belongs to a variance component
	  uint64 k = snpVCnums[m];
	  for (uint64 b = 0; b < B; b++) {
	    uint64 coeffStart = mPlus*B*D + b*D;
	    double mult = ((int) b == MCtrials ? 1 : -1.0/MCtrials) * vcXscale2s[k];
	    int curPar = k*D*(D+1)/2;
	    for (uint64 di = 0; di < D; di++)
	      for (uint64 dj = 0; dj <= di; dj++)
		grad(curPar++) += (di == dj ? 0.5 : 1.0) * mult * XtransVecsBlock[coeffStart+di] *
		  XtransVecsBlock[coeffStart+dj];
	  }
	}
      }
      
    }
    ALIGNED_FREE(XtransVecsBlock);
    ALIGNED_FREE(work);
    ALIGNED_FREE(snpCovCompVecBlock);

    // compute grad components for env variance parameters
    for (int t = 0; t <= MCtrials; t++) {
      double mult = t == MCtrials ? 1 : -1.0/MCtrials;
      int curPar = 0;
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++)
	  grad(curPar++) += (di == dj ? 0.5 : 1.0) * mult *
	    dotCovCompVecs(VinvyMultiCovCompVecs + t*DxNC + di*(Nstride+Cstride),
			   VinvyMultiCovCompVecs + t*DxNC + dj*(Nstride+Cstride));
    }
    
    return grad;
  }

  /**
   * computes Vmulti * X for a batch of ((N+C)D)-vectors X (each a stack of D (N+C)-vectors), where
   *     Vmulti = kron(VegXscale2s[0], I) + SUM_k=1^VCs kron(VegXscale2s[k], X_k*X_k')
   * and VegXscale2s = Veg .* [1 vcXscale2s[1]..vcXscale2s[VCs]]
   *
   * VmultiCovCompVecs: (out) B x D x (Nstride+Cstride)
   * xMultiCovCompVecs: (in) B x D x (Nstride+Cstride)
   * VegXscale2s: (in) (1+VCs) x DxD combined scale factors for easy application
   */
  void Bolt::multVmulti(double VmultiCovCompVecs[], const double xMultiCovCompVecs[],
			const uchar snpVCnums[],
			const vector <ublas::matrix <double> > &VegXscale2s, uint64 B) const {

    // note: algorithm is more efficient than from original multH:
    // for each block, it performs multXtrans, scales the results, and immediately does multX
    // this way, SNPs only need to be loaded once, and all VCs can be done at once

    uint64 D = VegXscale2s[0].size1();
    uint64 DxNC = D * (Nstride+Cstride);
    memcpy(VmultiCovCompVecs, xMultiCovCompVecs, B * DxNC * sizeof(VmultiCovCompVecs[0]));
    // initialize each answer vec Vmulti to X (NC x D) * Ve (D x D) for environment/noise VC
    for (uint64 b = 0; b < B; b++)
      for (uint64 nc = 0; nc < Nstride+Cstride; nc++)
	rightMultiT(VmultiCovCompVecs + b*DxNC + nc, D, Nstride+Cstride, VegXscale2s[0]);
    
    double *snpCovCompVecBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * (Nstride+Cstride));
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(omp_get_max_threads() * 256*sizeof(*work));
    double *XtransVecsBlock = ALIGNED_MALLOC_DOUBLES(mBlockMultX * B * D);

    // for each SNP block...
    for (uint64 m0 = 0; m0 < M; m0 += mBlockMultX) {

      // (1) load the SNPs in parallel into Xblock [Nstride+Cstride x block size]
      uint64 mBlockMultXCrop = std::min(M, m0+mBlockMultX) - m0;
#pragma omp parallel for
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (projMaskSnps[m] && snpVCnums[m]) // build snp vector + sign-flipped covar comps
	  buildMaskedSnpNegCovCompVec(snpCovCompVecBlock + mPlus * (Nstride+Cstride), m,
				      work + (omp_get_thread_num()<<8));
	else
	  memset(snpCovCompVecBlock + mPlus * (Nstride+Cstride), 0,
		 (Nstride+Cstride) * sizeof(snpCovCompVecBlock[0]));
      }

      // (2) multiply Xblock' [block size x Nstride+Cstride] * xMultiVecs [Nstride+Cstride x BxD]
      //              = XtransVecsBlock [block size x BxD]
      //     (note that Xblock' has NEG CCVecs while xVecs has POS CCVecs)
      {
#ifdef MEASURE_DGEMM
      //Timer timer;
      unsigned long long tsc = Timer::rdtsc();
#endif
	char TRANSA_ = 'T';
	char TRANSB_ = 'N';
	int M_ = B * D;
	int N_ = mBlockMultXCrop;
	int K_ = Nstride+Cstride;
	double ALPHA_ = 1.0;
	const double *A_ = xMultiCovCompVecs;
	int LDA_ = Nstride+Cstride;
	double *B_ = snpCovCompVecBlock;
	int LDB_ = Nstride+Cstride;
	double BETA_ = 0.0;
	double *C_ = XtransVecsBlock;
	int LDC_ = B * D;

	DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		    &BETA_, C_, &LDC_);
#ifdef MEASURE_DGEMM
      dgemmTicks += Timer::rdtsc() - tsc;
      //dgemmTicks += timer.update_time();
#endif
      }
      
      // (3) SNP-wise multiply XtransVecsBlock by appropriate VegXscale2 factors
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (snpVCnums[m]) { // SNP belongs to a variance component
	  uint64 k = snpVCnums[m];
	  for (uint64 b = 0; b < B; b++)
	    rightMultiT(XtransVecsBlock + mPlus*B*D + b*D, D, 1, VegXscale2s[k]);
	}
      }

      // note that Xblock was loaded with NEG CCVecs; now we need to sign-flip CCVecs
      for (uint64 mPlus = 0; mPlus < mBlockMultXCrop; mPlus++) {
	uint64 m = m0+mPlus;
	if (projMaskSnps[m] && snpVCnums[m]) // sign-flip covar comps
	  for (uint64 c = 0; c < Cstride; c++)
	    snpCovCompVecBlock[mPlus * (Nstride+Cstride) + Nstride + c] *= -1;
      }

      // (4) multiply Xblock [NCstride x block size] * scaled XtransVecsBlock [block size x B]
      //              = block's contribution to SUM_k (scale(k) X_k X_k') * xVecs [NCstride x B]
      //     directly accumulate results in answer Wmult [NCstride x B]
      {
#ifdef MEASURE_DGEMM
      //Timer timer;
      unsigned long long tsc = Timer::rdtsc();
#endif
	char TRANSA_ = 'N';
	char TRANSB_ = 'T';
	int M_ = Nstride+Cstride;
	int N_ = B * D;
	int K_ = mBlockMultXCrop;
	double ALPHA_ = 1.0;
	double *A_ = snpCovCompVecBlock;
	int LDA_ = Nstride+Cstride;
	const double *B_ = XtransVecsBlock;
	int LDB_ = B * D;
	double BETA_ = 1.0;
	double *C_ = VmultiCovCompVecs;
	int LDC_ = Nstride+Cstride;
	DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		    &BETA_, C_, &LDC_);
#ifdef MEASURE_DGEMM
      dgemmTicks += Timer::rdtsc() - tsc;
      //dgemmTicks += timer.update_time();
#endif
      }

    }
    ALIGNED_FREE(XtransVecsBlock);
    ALIGNED_FREE(work);
    ALIGNED_FREE(snpCovCompVecBlock);
  }

  /**
   * solves a batch of B equations
   *     Vmulti * vec(X) = b
   * where
   *     Vmulti = kron(Vegs[0], I) + SUM_k=1^VCs kron(Vegs[k], vcXscale2s[k] * X_k*X_k')
   * and vcXscale2s[k] = 1/("M_k" = sum(Xnorm2s[snpVCnums==k])/(Nused-Cindep))
   * (and as usual, projections of x, b, and columns of X are implicitly represented via covComps)
   *
   * xMultiCovCompVecs: (in/out) B x D x Nstride+Cstride
   * useStartVecs: use input xMultiCovCompVecs as initial guess for iteration
   * bMultiCovCompVecs: (in) B x D x Nstride+Cstride
   * Vegs: (in) (1+VCs) x DxD covariance parameters
   */
  void Bolt::conjGradSolveVmulti(double xMultiCovCompVecs[], bool useStartVecs,
				 const double bMultiCovCompVecs[], uint64 B,
				 const uchar snpVCnums[], const vector <double> &vcXscale2s,
				 const vector < ublas::matrix <double> > &Vegs, int maxIters,
				 double CGtol) const {

    int VCs = Vegs.size()-1;
    uint64 D = Vegs[0].size1();

    // combine Vegs and vcXscale2s into VegXscale2s
    vector < ublas::matrix <double> > VegXscale2s(1+VCs);
    VegXscale2s[0] = Vegs[0];
    for (int k = 1; k <= VCs; k++)
      VegXscale2s[k] = vcXscale2s[k] * Vegs[k];

#ifdef VERBOSE
    Timer timer;
    cout << "  Batch-solving " << B << " systems of equations using conjugate gradient iteration"
	 << endl;
#endif
#ifdef MEASURE_DGEMM
    unsigned long long tscStart = Timer::rdtsc();
    dgemmTicks = 0;
#endif

    const uint64 BxDxNC = B * D * (Nstride+Cstride);

    vector <double> r2orig(B), r2olds(B), r2news(B);
    computeMultiProjNorm2s(&r2orig[0], bMultiCovCompVecs, D, B);

    double *rMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES(BxDxNC);
    double *VmultiCovCompVecs = ALIGNED_MALLOC_DOUBLES(BxDxNC);
    if (useStartVecs) {
      multVmulti(VmultiCovCompVecs, xMultiCovCompVecs, snpVCnums, VegXscale2s, B); // V*x
      for (uint64 bdnc = 0; bdnc < BxDxNC; bdnc++)
	rMultiCovCompVecs[bdnc] = bMultiCovCompVecs[bdnc] - VmultiCovCompVecs[bdnc]; // r=b-V*x
      computeMultiProjNorm2s(&r2olds[0], rMultiCovCompVecs, D, B); // rsold=r'*r
    }
    else { // starting at x=0
      memset(xMultiCovCompVecs, 0, BxDxNC * sizeof(rMultiCovCompVecs[0]));
      memcpy(rMultiCovCompVecs, bMultiCovCompVecs, BxDxNC * sizeof(rMultiCovCompVecs[0]));
      r2olds = r2orig;
    }

    double *pMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES(BxDxNC);
    memcpy(pMultiCovCompVecs, rMultiCovCompVecs, BxDxNC * sizeof(pMultiCovCompVecs[0])); // p=r

    for (int iter = 0; iter < maxIters; iter++) {
      multVmulti(VmultiCovCompVecs, pMultiCovCompVecs, snpVCnums, VegXscale2s, B); // V*p

      for (uint64 bdnc = 0, b = 0; b < B; b++) {
	double *p = pMultiCovCompVecs + b * D*(Nstride+Cstride);
	double *Vp = VmultiCovCompVecs + b * D*(Nstride+Cstride);
	
	// alpha=rsold/(p'*Ap)
	double alpha = r2olds[b] / dotMultiCovCompVecs(p, Vp, D);

	for (uint64 dnc = 0; dnc < D*(Nstride+Cstride); dnc++, bdnc++) {
	  xMultiCovCompVecs[bdnc] += alpha * pMultiCovCompVecs[bdnc]; //x=x+alpha*p
	  rMultiCovCompVecs[bdnc] -= alpha * VmultiCovCompVecs[bdnc]; //r=r-alpha*Ap
	}
      }
      
      computeMultiProjNorm2s(&r2news[0], rMultiCovCompVecs, D, B); // rsnew=r'*r
      
#ifdef VERBOSE
      double min_rRatio = 1e9, max_rRatio = 0;
      for (uint64 b = 0; b < B; b++) {
	double rRatio = sqrt(r2news[b] / r2orig[b]);
	min_rRatio = std::min(rRatio, min_rRatio);
	max_rRatio = std::max(rRatio, max_rRatio);
      }

      vector <double> resNorm2s(B);
      computeMultiProjNorm2s(&resNorm2s[0], xMultiCovCompVecs, D, B);

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
	break;
      }

      for (uint64 bdnc = 0, b = 0; b < B; b++) {
	double r2ratio = r2news[b] / r2olds[b];
	for (uint64 dnc = 0; dnc < D*(Nstride+Cstride); dnc++, bdnc++) // p=r+rsnew/rsold*p
	  pMultiCovCompVecs[bdnc] = rMultiCovCompVecs[bdnc] + r2ratio * pMultiCovCompVecs[bdnc];
      }

      r2olds = r2news; // rsold=rsnew
    }
    
    ALIGNED_FREE(pMultiCovCompVecs);
    ALIGNED_FREE(VmultiCovCompVecs);
    ALIGNED_FREE(rMultiCovCompVecs);
#ifdef MEASURE_DGEMM
    double dgemmPct = 100 * dgemmTicks / (double) (Timer::rdtsc()-tscStart);
    printf("  Time breakdown: dgemm = %.1f%%, memory/overhead = %.1f%%\n", dgemmPct, 100-dgemmPct);
    fflush(stdout);
#endif
  }

  /**
   * creates yRandsData (MCtrials+1) MultiCovCompVecs as linear combinations of yEnvGenUnscaled
   * viewing yEnvGenUnscaled as (1+VCs) blocks of size (MCtrials+1) x D x (Nstride+Cstride),
   * yRandsData = lin. comb. of these blocks, where each NCxD chunk is right-mult by chol(Vegs)'
   * (note that the last rep is the data and just gets yEnvGenUnscaled(0,MCtrials,:) copied in)
   *
   * yRandsDataMultiCovCompVecs: (out) (MCtrials+1) x D x (Nstride+Cstride)
   * yEnvGenUnscaledMultiCovCompVecs: (in) (1+VCs) x (MCtrials+1) x D x (Nstride+Cstride)
   * 
   */
  void Bolt::combineEnvGenMultiCovCompVecs(double yRandsDataMultiCovCompVecs[],
					   const double yEnvGenUnscaledMultiCovCompVecs[],
					   const vector < ublas::matrix <double> > &Vegs,
					   int MCtrials) const {

    int VCs = Vegs.size()-1;
    uint64 D = Vegs[0].size1();
    vector < ublas::matrix <double> > cholVegs(1+VCs);
    for (int k = 0; k <= VCs; k++)
      cholVegs[k] = MatrixUtils::chol(Vegs[k]);
    uint64 DxNC = D * (Nstride+Cstride);
    uint64 blockSize = (MCtrials+1) * DxNC;
    memcpy(yRandsDataMultiCovCompVecs, yEnvGenUnscaledMultiCovCompVecs, // copy in env terms
	   blockSize * sizeof(yRandsDataMultiCovCompVecs[0]));
    vector <double> tmpD(D);
    for (int t = 0; t < MCtrials; t++)
      for (uint64 nc = 0; nc < Nstride+Cstride; nc++) {
	// apply chol' to env terms currently in yRands (but not data rep)
	rightMultiT(yRandsDataMultiCovCompVecs + t*DxNC + nc, D, Nstride+Cstride, cholVegs[0]);
	// apply chol' to gen terms and accumulate in yRands
	for (int k = 1; k <= VCs; k++) {	  
	  const double *yGenUnscaledMultiCCVecs_k = yEnvGenUnscaledMultiCovCompVecs + k*blockSize;
	  for (uint64 d = 0; d < D; d++)
	    tmpD[d] = yGenUnscaledMultiCCVecs_k[t*DxNC + d*(Nstride+Cstride) + nc];
	  rightMultiT(&tmpD[0], D, 1, cholVegs[k]);
	  for (uint64 d = 0; d < D; d++)
	    yRandsDataMultiCovCompVecs[t*DxNC + d*(Nstride+Cstride) + nc] += tmpD[d];
	}
      }
  }

  /**
   * TODO: update comments (basically the same as single-trait version but with MCtrials x D rands)
   *
   * generates 1+VCs blocks of MCtrials+1 (rand+data) component vecs for building MCreml phenotypes
   * later, blocks will be combined with coeffs 1, rho_1, ..., rho_VCs
   * components are pre-scaled only with sqrt(vcXscale2s) (~1/sqrt(M_k)), putting Xs on same scale
   *
   * yEnvGenUnscaledCovCompVecs: (out) (1+VCs) x (MCtrials+1) x (Nstride+Cstride)
   *                                   env gen    rands data
   *                         data rep:  y  000
   *
   * - projecting out covariates: implicitly done by covComps
   * - applying maskIndivs: automatically done to snps (=> Gen component) by buildMaskedSnpVector
   *                        needs to be applied to EnvUnscaled component
   * - data layout: 1+VCs batches of {MCtrials rand reps + 1 data rep}
   *   - first batch holds Env components
   *   - remaining VCs batches hold Gen components
   *   - in each batch:
   *     - 0..MCtrials-1: Env, Gen components of random phenotypes
   *     - MCtrials: Env = pheno from data; Gen = 0
   *
   * pheno: (in) real phenotype (data rep), possibly of size N or zero-filled beyond (no covComps)
   * snpVCnums: (in) M-vector of assignments of SNPs to VCs (0 -> ignore; 1..VCs -> var comps)
   * VCs: (in) number of non-identity VCs
   * vcXscale2s: (in) (VCs+1)-vector of squared scale factors that normalize X's (ignore 0th entry)
   *
   * return: phenotype normalizations and correlations
   */
  ublas::matrix <double> Bolt::genUnscaledMultiCovCompVecs
  (double yEnvGenUnscaledMultiCovCompVecs[], const vector < vector <double> > &phenos,
   const uchar snpVCnums[], int VCs, const vector <double> &vcXscale2s, int MCtrials, int seed)
    const {

    boost::mt19937 rng(seed+54321);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >
      randn(rng, boost::normal_distribution<>(0.0, 1.0));

    uint64 D = phenos.size();
    uint64 DxNC = D * (Nstride+Cstride);

    // return matrix: phenotype normalizations and corrs
    ublas::matrix <double> phenoNormsCorrs = ublas::zero_matrix <double> (D, D);

    // zero-initialize all vectors
    memset(yEnvGenUnscaledMultiCovCompVecs, 0,
	   (1+VCs) * (MCtrials+1) * DxNC * sizeof(yEnvGenUnscaledMultiCovCompVecs[0]));

    // put pheno in Env (first=0 of 1+VCs) block, MCtrials (last=MCtrials of MCtrials+1) DxNC vec
    double *phenoCovCompVecs = yEnvGenUnscaledMultiCovCompVecs + MCtrials * DxNC;
    for (uint64 d = 0; d < D; d++) {
      double *phenoCovCompVecs_d = phenoCovCompVecs + d*(Nstride+Cstride);
      memcpy(phenoCovCompVecs_d, &phenos[d][0], phenos[d].size() * sizeof(phenoCovCompVecs[0]));
      covBasis.applyMaskIndivs(phenoCovCompVecs_d);
      covBasis.computeCindepComponents(phenoCovCompVecs_d + Nstride, phenoCovCompVecs_d);
      // rescale phenos to have norm approximately 1
      double scale = sqrt((Nused-Cindep) / computeProjNorm2(phenoCovCompVecs_d));
      phenoNormsCorrs(d, d) = 1/scale;
      for (uint nc = 0; nc < Nstride+Cstride; nc++)
	phenoCovCompVecs_d[nc] *= scale;
    }
    for (uint64 i = 0; i < D; i++)
      for (uint64 j = i+1; j < D; j++)
	phenoNormsCorrs(i, j) = phenoNormsCorrs(j, i) =
	  dotCovCompVecs(phenoCovCompVecs + i*(Nstride+Cstride),
			 phenoCovCompVecs + j*(Nstride+Cstride)) / (Nused-Cindep);

    // generate yGen: VCs x (MCtrials+1) x DxNC with rand betas generated on-the-fly!
    double *snpCovCompVec = ALIGNED_MALLOC_DOUBLES(Nstride+Cstride);
    double (*work)[4] = (double (*)[4]) ALIGNED_MALLOC(256*sizeof(*work));
    double *betas_m = ALIGNED_MALLOC_DOUBLES(MCtrials*D);
    for (uint64 m = 0; m < M; m++)
      if (projMaskSnps[m] && snpVCnums[m]) {
	// loop through SNPs; load just 1 (Nstride+Cstride) SNP at a time
	buildMaskedSnpCovCompVec(snpCovCompVec, m, work);
	for (int t = 0; t < MCtrials * (int) D; t++) // generate MCtrials*D random betas
	  betas_m[t] = randn();
	// DGER to update MCtrials x D x (Nstride+Cstride) block *for appropriate VC*
	uint64 k = snpVCnums[m];
	double *yGenBlock = yEnvGenUnscaledMultiCovCompVecs + k * (MCtrials+1) * DxNC;
	// update yGenBlock: add betas_m * snpCovCompVec (DGER)
	{
	  int M_ = Nstride+Cstride;
	  int N_ = MCtrials*D;
	  double ALPHA_ = sqrt(vcXscale2s[k]); // normalize 1/sqrt("M"=sum(Xnorm2s)/(Nused-Cindep))
	  double *X_ = snpCovCompVec;
	  int INCX_ = 1;
	  double *Y_ = betas_m;
	  int INCY_ = 1;
	  double *A_ = yGenBlock;
	  int LDA_ = Nstride+Cstride;
	  DGER_MACRO(&M_, &N_, &ALPHA_, X_, &INCX_, Y_, &INCY_, A_, &LDA_);
	}
      }
    ALIGNED_FREE(betas_m);
    ALIGNED_FREE(work);
    ALIGNED_FREE(snpCovCompVec);

    // generate yEnv: first MCtrials x (Nstride+Cstride) block of output array
    // (last 1 x (Nstride+Cstride) is real data phenotype)
    for (int t = 0; t < MCtrials; t++) {
      // EnvUnscaled: epsCovCompVec <- N randn, after the following processing...
      // - mask out maskIndivs (=> norm2 ~ Nused)
      // - compute covComps: implicitly project out covars (=> norm2-SUM(comps2) ~ Nused-Cindep)
      for (uint64 d = 0; d < D; d++) {
	double *randnEpsCovCompVec = yEnvGenUnscaledMultiCovCompVecs + t*DxNC+d*(Nstride+Cstride);
	for (uint64 n = 0; n < Nstride; n++)
	  if (maskIndivs[n])
	    randnEpsCovCompVec[n] = randn();
	// no need to zero out components after Cindep: already 0-initialized
	covBasis.computeCindepComponents(randnEpsCovCompVec + Nstride, randnEpsCovCompVec);
      }
    }

    return phenoNormsCorrs;
  }

  void Bolt::updateVegsAI(vector < ublas::matrix <double> > &Vegs, const uchar snpVCnums[],
			  const vector <double> &vcXscale2s,
			  const double yRandsDataMultiCovCompVecs[], int MCtrials, int CGmaxIters,
			  double CGtol) const {
    
    int VCs = Vegs.size()-1;
    uint64 D = Vegs[0].size1();
    uint64 DxNC = D * (Nstride+Cstride);

    double *VinvyRandsDataMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES((MCtrials+1) * DxNC);

    conjGradSolveVmulti(VinvyRandsDataMultiCovCompVecs, false, yRandsDataMultiCovCompVecs,
			MCtrials+1, snpVCnums, vcXscale2s, Vegs, CGmaxIters, CGtol);

    // compute gradient
    vector < ublas::matrix <double> > identityVegs(1+VCs, ublas::identity_matrix <double> (D));
    ublas::vector <double> grad = updateVegs(identityVegs, VinvyRandsDataMultiCovCompVecs,
					     snpVCnums, vcXscale2s, MCtrials);
    cout << "grad" << grad << endl;

    // compute AI matrix

    const double *VinvyMultiCovCompVecs = VinvyRandsDataMultiCovCompVecs + MCtrials * DxNC;
    
    double *ThetasVinvyMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES(VCs * DxNC);

    multThetaMinusIs(ThetasVinvyMultiCovCompVecs, VinvyMultiCovCompVecs, snpVCnums, vcXscale2s,
		     D, 0);
    
    int numPars = (1+VCs) * D*(D+1)/2;
    double *dVdparsVinvyMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES(numPars * DxNC);

    memset(dVdparsVinvyMultiCovCompVecs, 0, numPars*DxNC*sizeof(dVdparsVinvyMultiCovCompVecs[0]));
    int curPar = 0;
    for (int k = 0; k <= VCs; k++) { // populate dVdparsVinvyMultiCovCompVecs with computed vecs
      const double *ThetasVinvyMultiCCVecs_k = k == 0 ? VinvyMultiCovCompVecs :
	ThetasVinvyMultiCovCompVecs + (k-1) * DxNC;
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++) {
	  memcpy(dVdparsVinvyMultiCovCompVecs + curPar * DxNC + dj*(Nstride+Cstride),
		 ThetasVinvyMultiCCVecs_k + di*(Nstride+Cstride),
		 (Nstride+Cstride)*sizeof(ThetasVinvyMultiCCVecs_k[0]));
	  if (di != dj)
	    memcpy(dVdparsVinvyMultiCovCompVecs + curPar * DxNC + di*(Nstride+Cstride),
		   ThetasVinvyMultiCCVecs_k + dj*(Nstride+Cstride),
		   (Nstride+Cstride)*sizeof(ThetasVinvyMultiCCVecs_k[0]));
	  curPar++;
	}
    }

    double *VinvdVdparsVinvyMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES(numPars * DxNC);

    conjGradSolveVmulti(VinvdVdparsVinvyMultiCovCompVecs, false, dVdparsVinvyMultiCovCompVecs,
			numPars, snpVCnums, vcXscale2s, Vegs, CGmaxIters, CGtol);

    ublas::matrix <double> AI = ublas::zero_matrix <double> (numPars, numPars);
   
    for (int pi = 0; pi < numPars; pi++)
      for (int pj = 0; pj < numPars; pj++)
	AI(pi, pj) = pj < pi ? AI(pj, pi) :
	  -0.5 * dotMultiCovCompVecs(VinvdVdparsVinvyMultiCovCompVecs + pi * DxNC,
				     dVdparsVinvyMultiCovCompVecs + pj * DxNC, D);
    cout << "AI" << AI << endl;

    ublas::vector <double> step = -MatrixUtils::linSolve(AI, grad);
    cout << "step" << step << endl;
    
    curPar = 0;
    for (int k = 0; k <= VCs; k++)
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++) {
	  Vegs[k](di, dj) += step(curPar++);
	  Vegs[k](dj, di) = Vegs[k](di, dj);
	}

    ALIGNED_FREE(VinvdVdparsVinvyMultiCovCompVecs);
    ALIGNED_FREE(dVdparsVinvyMultiCovCompVecs);
    ALIGNED_FREE(ThetasVinvyMultiCovCompVecs);
    ALIGNED_FREE(VinvyRandsDataMultiCovCompVecs);
  }

  ublas::matrix <double> Bolt::computeAI(const vector < ublas::matrix <double> > &Vegs,
					 const double VinvyMultiCovCompVecs[],
					 const uchar snpVCnums[],
					 const vector <double> &vcXscale2s, int CGmaxIters,
					 double CGtol) const {
    
    int VCs = Vegs.size()-1;
    uint64 D = Vegs[0].size1();
    uint64 DxNC = D * (Nstride+Cstride);

    double *ThetasVinvyMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES(VCs * DxNC);

    multThetaMinusIs(ThetasVinvyMultiCovCompVecs, VinvyMultiCovCompVecs, snpVCnums, vcXscale2s,
		     D, 0);
    
    int numPars = (1+VCs) * D*(D+1)/2;
    double *dVdparsVinvyMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES(numPars * DxNC);

    memset(dVdparsVinvyMultiCovCompVecs, 0, numPars*DxNC*sizeof(dVdparsVinvyMultiCovCompVecs[0]));
    int curPar = 0;
    for (int k = 0; k <= VCs; k++) { // populate dVdparsVinvyMultiCovCompVecs with computed vecs
      const double *ThetasVinvyMultiCCVecs_k = k == 0 ? VinvyMultiCovCompVecs :
	ThetasVinvyMultiCovCompVecs + (k-1) * DxNC;
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++) {
	  memcpy(dVdparsVinvyMultiCovCompVecs + curPar * DxNC + dj*(Nstride+Cstride),
		 ThetasVinvyMultiCCVecs_k + di*(Nstride+Cstride),
		 (Nstride+Cstride)*sizeof(ThetasVinvyMultiCCVecs_k[0]));
	  if (di != dj)
	    memcpy(dVdparsVinvyMultiCovCompVecs + curPar * DxNC + di*(Nstride+Cstride),
		   ThetasVinvyMultiCCVecs_k + dj*(Nstride+Cstride),
		   (Nstride+Cstride)*sizeof(ThetasVinvyMultiCCVecs_k[0]));
	  curPar++;
	}
    }

    double *VinvdVdparsVinvyMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES(numPars * DxNC);

    conjGradSolveVmulti(VinvdVdparsVinvyMultiCovCompVecs, false, dVdparsVinvyMultiCovCompVecs,
			numPars, snpVCnums, vcXscale2s, Vegs, CGmaxIters, CGtol);

    ublas::matrix <double> AI = ublas::zero_matrix <double> (numPars, numPars);
   
    for (int pi = 0; pi < numPars; pi++)
      for (int pj = 0; pj < numPars; pj++)
	AI(pi, pj) = pj < pi ? AI(pj, pi) :
	  0.5 * dotMultiCovCompVecs(VinvdVdparsVinvyMultiCovCompVecs + pi * DxNC,
				    dVdparsVinvyMultiCovCompVecs + pj * DxNC, D);

    ALIGNED_FREE(VinvdVdparsVinvyMultiCovCompVecs);
    ALIGNED_FREE(dVdparsVinvyMultiCovCompVecs);
    ALIGNED_FREE(ThetasVinvyMultiCovCompVecs);

    return AI;
  }

  /**
   * pheno: (in) real phenotype (data rep), possibly of size N or zero-filled beyond (no covComps)
   * snpVCnums: (in) M-vector of assignments of SNPs to VCs (0 -> ignore; 1..VCs -> var comps)
   */
  void Bolt::remlAI(vector < ublas::matrix <double> > &Vegs, bool usePhenoCorrs,
		    const vector < vector <double> > &phenos, const uchar snpVCnums[],
		    int MCtrialsCoarse, int MCtrialsFine, int CGmaxIters, double CGtol,
		    int seed) const {

    // determine number of VCs and scale factors vcXscale2s = 1 / "M_k":
    // view each X_k as X_k * 1/sqrt("M_k" = sum(Xnorm2s)/(Nused-Cindep));
    // then all 1+VCs var comps (inc. identity) are on same footing
    int VCs = 0; vector <double> vcXscale2s(1, 1);
    for (uint64 m = 0; m < M; m++) {
      if (projMaskSnps[m] && snpVCnums[m] > VCs) {
	VCs = snpVCnums[m];
	vcXscale2s.resize(VCs+1);
      }
      if (projMaskSnps[m] && snpVCnums[m])
	vcXscale2s[snpVCnums[m]] += Xnorm2s[m];
    }
    for (int k = 1; k <= VCs; k++)
      vcXscale2s[k] = (Nused-Cindep)/vcXscale2s[k];

    if (VCs != (int) Vegs.size() - 1) {
      cerr << "ERROR: # of VCs represented in non-masked SNPs does not match # in model" << endl;
      cerr << "       Did a variance component lose all of its SNPs during Bolt QC?" << endl;
      exit(1);
    }

    uint64 D = phenos.size();
    uint64 DxNC = D * (Nstride+Cstride);
    int numPars = (1+VCs) * D*(D+1)/2;

    ublas::matrix <double> AI;
    ublas::matrix <double> phenoNormsCorrs;
    int MCtrials;
    double tolLL;

    for (int phase = 0; phase < 2; phase++) {
      cout << endl
	   << "==============================================================================="
	   << endl << endl;
      if (phase == 0) {
	MCtrials = MCtrialsCoarse;
	tolLL = 1e-2;
	cout << "Stochastic REML optimization with MCtrials = " << MCtrials << endl << endl;
      }
      else {
	if (MCtrialsFine <= MCtrialsCoarse)
	  break;
	MCtrials = MCtrialsFine;
	tolLL = 1e-4;
	cout << "Refining REML optimization with MCtrials = " << MCtrials << endl << endl;
      }

    // generate env and gen (1 gen per VC) components for rand, data phenotype vectors
    double *yEnvGenUnscaledMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES((1+VCs)*(MCtrials+1) * DxNC);
    phenoNormsCorrs = genUnscaledMultiCovCompVecs(yEnvGenUnscaledMultiCovCompVecs, phenos,
						  snpVCnums, VCs, vcXscale2s, MCtrials, seed);
    cout << "phenoNormsCorrs" << phenoNormsCorrs << endl;
    if (phase == 0 && usePhenoCorrs)
      for (int k = 0; k <= VCs; k++)
	for (uint64 di = 0; di < D; di++)
	  for (uint64 dj = 0; dj < di; dj++)
	    Vegs[k](di, dj) = Vegs[k](dj, di) =
	      sqrt(Vegs[k](di, di) * Vegs[k](dj, dj)) * phenoNormsCorrs(di, dj);

    double *yRandsDataMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES((MCtrials+1) * DxNC);
    double *VinvyRandsDataMultiCovCompVecs = ALIGNED_MALLOC_DOUBLES((MCtrials+1) * DxNC);
    const double *VinvyMultiCovCompVecs = VinvyRandsDataMultiCovCompVecs + MCtrials * DxNC;    

      /* EM step
      combineEnvGenMultiCovCompVecs(yRandsDataMultiCovCompVecs, yEnvGenUnscaledMultiCovCompVecs,
  				    Vegs, MCtrials);
      conjGradSolveVmulti(VinvyRandsDataMultiCovCompVecs, false, yRandsDataMultiCovCompVecs,
			  MCtrials+1, snpVCnums, vcXscale2s, Vegs, CGmaxIters, CGtol);
      updateVegs(Vegs, VinvyRandsDataMultiCovCompVecs, snpVCnums, vcXscale2s, MCtrials);
      */
    
    cout << "Initial variance parameter guesses:" << endl;
    for (int k = 0; k <= VCs; k++)
      cout << "Vegs[" << k << "]" << Vegs[k] << endl;
    cout << endl;
    cout << "Performing initial gradient evaluation" << endl;
    // compute gradient
    combineEnvGenMultiCovCompVecs(yRandsDataMultiCovCompVecs, yEnvGenUnscaledMultiCovCompVecs,
				  Vegs, MCtrials);
    conjGradSolveVmulti(VinvyRandsDataMultiCovCompVecs, false, yRandsDataMultiCovCompVecs,
			MCtrials+1, snpVCnums, vcXscale2s, Vegs, CGmaxIters, CGtol);

    ublas::vector <double> grad = computeMCgrad(VinvyRandsDataMultiCovCompVecs, D, snpVCnums, VCs,
						vcXscale2s, MCtrials);
    cout << "grad" << grad << endl << endl;

    const double eta1 = 1e-4, eta2 = 0.99;
    const double alpha1 = 0.25, alpha2 = 3.5;
    double Delta = 1e100; // initialize step norm bound to large

    const int AImaxIters = 20;
    for (int iter = 0; iter < AImaxIters; iter++) {
      cout << "-------------------------------------------------------------------------------"
	   << endl << endl;
      cout << "Start ITER " << (iter+1) << ": computing AI matrix" << endl;
      AI = computeAI(Vegs, VinvyMultiCovCompVecs, snpVCnums, vcXscale2s, CGmaxIters, CGtol);
      //cout << "AI" << AI << endl;

      double dLL = -1;
      int att, maxAttempts = 5;
      bool converged = false;
      for (att = 1; att <= maxAttempts; att++) {
	double dLLpred; ublas::vector <double> p;
	vector < ublas::matrix <double> > optVegs =
	  NonlinearOptMulti::constrainedNR(dLLpred, p, Vegs, grad, AI, Delta);
	cout << endl << "Constrained Newton-Raphson optimized variance parameters:" << endl;
	for (int k = 0; k <= VCs; k++)
	  cout << "optVegs[" << k << "]" << optVegs[k] << endl;
	cout << endl;
	cout << "Predicted change in log likelihood: " << dLLpred << endl;
	if (dLLpred < tolLL) {
	  cout << "AI iteration converged: predicted change in log likelihood < tol = " << tolLL
	       << endl;
	  Vegs = optVegs;
	  converged = true;
	  break;
	}

	cout << endl << "Computing actual (approximate) change in log likelihood" << endl;
	combineEnvGenMultiCovCompVecs(yRandsDataMultiCovCompVecs, yEnvGenUnscaledMultiCovCompVecs,
				      optVegs, MCtrials);
	conjGradSolveVmulti(VinvyRandsDataMultiCovCompVecs, false, yRandsDataMultiCovCompVecs,
			    MCtrials+1, snpVCnums, vcXscale2s, optVegs, CGmaxIters, CGtol);
	ublas::vector <double> optGrad = computeMCgrad(VinvyRandsDataMultiCovCompVecs, D,
	                                               snpVCnums, VCs, vcXscale2s, MCtrials);
	cout << "grad" << optGrad << endl;
	dLL = ublas::inner_prod(p, 0.5*(grad+optGrad));
	cout << endl << "Approximate change in log likelihood: " << dLL
	     << " (attempt " << att << ")" << endl;

	double rho = dLL / dLLpred;
	if (ublas::norm_2(optGrad) > 2*ublas::norm_2(grad)) {
	  rho = -1;
	  cout << "Large increase in grad norm: dangerous model deviation?  Setting rho=-1"
	       << endl;
	}

	cout << "rho (approximate / predicted change in LL) = " << rho << endl;
	cout << "Old trust region radius: " << Delta << endl;

	// update trust region radius
	ublas::vector <double> Dp = p; // scale step coordinates using diagonal of AI matrix
	for (int par = 0; par < numPars; par++)
	  Dp(par) *= AI(par, par);

	if (rho < eta1) // bad step: reduce trust region
	  Delta = alpha1 * ublas::norm_2(Dp);
	else if (rho < eta2) // ok step: do nothing
	  ;
	else // great step: expand trust region
	  Delta = std::max(Delta, alpha2 * ublas::norm_2(Dp));
	cout << "New trust region radius: " << Delta << endl;

	if (rho > eta1) { // accept step and exit inner loop
	  cout << "Accepted step" << endl;
	  Vegs = optVegs;
	  grad = optGrad;
	  break;
	}
	else {
	  cout << "Rejected step" << endl;
	}
      }

      if (converged)
	break;
      else if (dLL < 0) {
	cerr << "WARNING: Failed to accept step in " << maxAttempts << " attempts" << endl;
	cerr << "         Stopping AI iteration, but optimization may not have converged" << endl;
	break;
      }
      else {
	cout << endl << "End ITER " << (iter+1) << endl;
	for (int k = 0; k <= VCs; k++)
	  cout << "Vegs[" << k << "]" << Vegs[k] << endl;
	cout << endl;
      }
    }

    ALIGNED_FREE(VinvyRandsDataMultiCovCompVecs);
    ALIGNED_FREE(yRandsDataMultiCovCompVecs);
    ALIGNED_FREE(yEnvGenUnscaledMultiCovCompVecs);

    }

    cout << endl;
    ublas::matrix <double> AIinv = MatrixUtils::invert(AI);
    cout << "AIinv" << AIinv << endl << endl;

    int curPar = 0;
    for (int k = 0; k <= VCs; k++) {
      cout << "Variance component " << k << ": " << Vegs[k] << endl;
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++) {
	  printf("  entry (%d,%d): %.6f (%.6f)", (int) dj+1, (int) di+1, Vegs[k](di, dj),
		 sqrt(AIinv(curPar, curPar)));
	  if (di != dj)
	    printf("   corr (%d,%d): %.6f", (int) dj+1, (int) di+1,
		   Vegs[k](di, dj) / sqrt(Vegs[k](di, di) * Vegs[k](dj, dj)));
	  cout << endl;
	  curPar++;
	}
    }
    cout << endl;

    // apply coordinate transformation to h2, r_g; compute SEs
    
    double sigma2s[D], h2rgs[1+VCs][D][D]; // transformed pars
    int par[1+VCs][D][D]; // indices of parameters in [0..numPars)
    ublas::matrix <double> J = ublas::zero_matrix <double> (numPars, numPars);

    // transform off-diagonals from covariances to correlations (to get SEs on corrs)
    curPar = 0;
    for (int k = 0; k <= VCs; k++)
      for (uint64 i = 0; i < D; i++)
	for (uint64 j = 0; j <= i; j++) {
	  // to get point estimates of gen corrs, just divide by sqrt prod
	  h2rgs[k][i][j] = i == j ? Vegs[k](i, j) :
	    Vegs[k](i, j) / sqrt(Vegs[k](i, i) * Vegs[k](j, j));	  
	  par[k][i][j] = curPar++;
	}
    for (int k = 0; k <= VCs; k++)
      for (uint64 i = 0; i < D; i++)
	for (uint64 j = 0; j <= i; j++) {
	  if (i == j)
	    J(par[k][i][i], par[k][i][i]) = 1;
	  else {
	    J(par[k][i][j], par[k][i][i]) = Vegs[k](i, j) / (2*Vegs[k](i, i));
	    J(par[k][i][j], par[k][i][j]) = Vegs[k](i, j) / (h2rgs[k][i][j]);
	    J(par[k][i][j], par[k][j][j]) = Vegs[k](i, j) / (2*Vegs[k](j, j));
	  }
	}
    ublas::matrix <double> rgAI =
      ublas::prod(ublas::matrix <double> (ublas::prod(ublas::trans(J), AI)), J);

    // transform variance coords to: (sigma2 scale parameter, h2_1, h2_2, ..., h2_VCs)
    for (int i = 0; i < (int) D; i++) {
      // to get point estimates of sigma2 for each trait, sum raw per-VC sigma2s over VCs
      sigma2s[i] = 0;
      for (int k = 0; k <= VCs; k++)
	sigma2s[i] += Vegs[k](i, i);
      // to get point estimate h2s (including env), take fractions (raw sigma2s over VCs) / sum
      for (int k = 0; k <= VCs; k++)
	h2rgs[k][i][i] = Vegs[k](i, i) / sigma2s[i];
    }
      
    ublas::matrix <double> h2rgAIinv[2]; h2rgAIinv[0] = h2rgAIinv[1] = rgAI;
    // leave out one k (VC): h2s for all other VCs, 1-sum(h2s) for left-out kOut
    // kOut = 0: get SEs for sigma2, h2 (for all traits), r_g (for all trait pairs)
    // kOut = 1: get SEs for environment/noise h2
    for (int kOut = 0; kOut < 2; kOut++) {
      for (int i = 0; i < (int) D; i++) { // apply transform to parameters for each trait in turn
	J = ublas::identity_matrix <double> (numPars);
	for (int k = 0; k <= VCs; k++) {
	  if (k != kOut) {
	    J(par[kOut][i][i], par[k][i][i]) = -sigma2s[i];
	    J(par[k][i][i], par[k][i][i]) = sigma2s[i];
	  }
	  J(par[k][i][i], par[kOut][i][i]) = h2rgs[k][i][i];
	}
	// note: these Js don't affect r_g coords
	h2rgAIinv[kOut] =
	  ublas::prod(ublas::matrix <double> (ublas::prod(ublas::trans(J), h2rgAIinv[kOut])), J);
      }
      h2rgAIinv[kOut] = MatrixUtils::invert(h2rgAIinv[kOut]);
    }

    for (int i = 0; i < (int) D; i++)
      printf("Phenotype %d variance sigma2: %f (%f)\n", i+1,
	     sigma2s[i] * NumericUtils::sq(phenoNormsCorrs(i, i)),
	     sqrt(h2rgAIinv[0](i, i)) * NumericUtils::sq(phenoNormsCorrs(i, i)));
    cout << endl;

    const vector <string> &vcNames = snpData.getVCnames();

    curPar = 0;
    for (int k = 0; k <= VCs; k++) {
      cout << "Variance component " << k << ": ";
      if (k == 0) cout << " (environment/noise)" << endl;
      else cout << " \"" << vcNames[k] << "\"" << endl;
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++) {
	  if (di == dj)
	    printf("  h2%c (%d,%d): %.6f (%.6f)", k==0?'e':'g', (int) dj+1, (int) di+1,
		   h2rgs[k][di][dj], sqrt((1+1.0/MCtrials) * h2rgAIinv[k==0](curPar, curPar)));
	  else
	    printf("  %s corr (%d,%d): %.6f (%.6f)", k==0?"resid":"gen",
		   (int) dj+1, (int) di+1, h2rgs[k][di][dj],
		   sqrt((1+1.0/MCtrials) * h2rgAIinv[0](curPar, curPar)));
	  cout << endl;
	  curPar++;
	}
    }

      /*
      vector <int> digits(VCs);
      for (int v = 0; v < VCs; v++)
	digits[v] = std::min(10, std::max(2, (int) -log10(std::max(evalData.xSEs(v), 1e-10)) + 2));
      int maxDigits = *std::max_element(digits.begin(), digits.end());
      for (int v = 0; v < VCs; v++) {
	char format[100];

	char h2buf[100];
	sprintf(format, "%%.%df", digits[v]);
	sprintf(h2buf, format, x[v]);

	char h2SEbuf[100];
	sprintf(format, "(%%.%df)", digits[v]);
	sprintf(h2SEbuf, format, evalData.xSEs(v));

	sprintf(format, "  Variance component %%d:   %%-%ds %%-%ds", maxDigits+2, maxDigits+4);
	printf(format, v+1, h2buf, h2SEbuf);
	cout << "   \"" << snpData.getVCnames()[v+1] << "\"" << endl;
      }
      */

    cout << endl;
  }

}
