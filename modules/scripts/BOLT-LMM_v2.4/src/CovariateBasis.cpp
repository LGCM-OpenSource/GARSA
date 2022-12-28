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
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <utility>
#include <boost/utility.hpp>

#include "LapackConst.hpp"
#include "Types.hpp"
#include "DataMatrix.hpp"
#include "MemoryUtils.hpp"
#include "CovariateBasis.hpp"

namespace LMM {

  using std::vector;
  using std::string;
  using std::pair;
  using std::cout;
  using std::cerr;
  using std::endl;

  const double CovariateBasis::MIN_REL_SINGULAR_VALUE = 1e-8;

  /*
  // default init: use all covariates in covarDataT as quantitative... probably a bad idea
  CovariateBasis::CovariateBasis(const DataMatrix &covarDataT, const double _maskIndivs[],
				 int covarMaxLevels) {
    vector < pair <string, DataMatrix::ValueType> > covars;
    for (uint64 r = 0; r < covarDataT.rowNames.size(); r++)
      covars.push_back(std::make_pair(covarDataT.rowNames[r], DataMatrix::QUANTITATIVE));
    init(covarDataT, _maskIndivs, covars, covarMaxLevels);
  }
  */

  CovariateBasis::CovariateBasis(const DataMatrix &covarDataT, const double _maskIndivs[],
				 const vector < pair <string, DataMatrix::ValueType> > &covars,
				 int covarMaxLevels, bool covarUseMissingIndic) {
    init(covarDataT, _maskIndivs, covars, covarMaxLevels, covarUseMissingIndic);
  }

  /**
   * input:
   * - covarDataT is assumed to contain the all-1s vector as its first row (indexed 0)
   * - _maskIndivs has dimension >= covarDataT.ncols = N (if > N, additional cols are masked)
   *   - assumed to be a subset of maskIndivs from original SnpData instance
   *   - (presumably obtained by using SnpData.writeMaskIndivs and taking a subset)
   * action:
   * - copies _maskIndivs into member maskIndivs (and optionally performs missing masking)
   * - builds masked copy of selected covariates (from covarDataT) in maskedCovars[C x Nstride]
   *   - (possible later use: DGELS to get least squares coeffs wrt original covars)
   * - mean-fills missing covariate values (using mean of non-masked, non-missing)
   * - computes SVD; stores in basis[C x Nstride] and sets Cindep (warns if < C)
   */
  void CovariateBasis::init(const DataMatrix &covarDataT, const double _maskIndivs[],
			    vector < pair <string, DataMatrix::ValueType> > covars,
			    int covarMaxLevels, bool covarUseMissingIndic) {

    // all-1s vector must always be included
    if (std::find(covars.begin(), covars.end(),
		  std::make_pair(DataMatrix::ALL_ONES_NAME, DataMatrix::QUANTITATIVE))
	== covars.end()) {
      cerr << "NOTE: Using all-1s vector (constant term) in addition to specified covariates"
	   << endl;
      covars.push_back(std::make_pair(DataMatrix::ALL_ONES_NAME, DataMatrix::QUANTITATIVE));
    }

    // allocate maskIndivs; temporarily store missingness status
    Nstride = covarDataT.ncols;
    maskIndivs = ALIGNED_MALLOC_DOUBLES(Nstride);
    for (uint64 n = 0; n < Nstride; n++) maskIndivs[n] = 1;

    // select covariates to use; error-check covars
    vector < std::pair <uint64, string> > covarIndLevelPairs; // levels if categorical; o/w empty
    const string QTVE_MISSING_INDICATOR = "QTVE_MISSING_INDICATOR";
    for (vector < pair <string, DataMatrix::ValueType> >::const_iterator it = covars.begin();
	 it != covars.end(); it++) {
      bool found = false;
      for (uint64 cData = 0; cData < covarDataT.nrows; cData++)
	if (covarDataT.rowNames[cData] == it->first) {
	  if (it->second == DataMatrix::QUANTITATIVE) {
	    // update temp maskIndivs with missingness
	    for (uint64 n = 0; n < Nstride; n++)
	      if (covarDataT.getEntryDbl(cData, n) == covarDataT.missing_key_dbl)
		maskIndivs[n] = 0;

	    cout << "    Using quantitative covariate: " << covarDataT.rowNames[cData] << endl;
	    covarIndLevelPairs.push_back(std::make_pair(cData, ""));
	    if (covarUseMissingIndic && it->first != DataMatrix::ALL_ONES_NAME) {
	      cout << "     (adding missing indicator: " << covarDataT.rowNames[cData] << ")"
		   << endl;
	      covarIndLevelPairs.push_back(std::make_pair(cData, QTVE_MISSING_INDICATOR));
	    }
	  }
	  else {
	    // update temp maskIndivs with missingness
	    for (uint64 n = 0; n < Nstride; n++)
	      if (covarDataT.isMissing(covarDataT.data[cData][n]))
		maskIndivs[n] = 0;

	    // note: missing indicator already included for categorical covars
	    for (std::set <string>::iterator levelsIter = covarDataT.rowUniqLevels[cData].begin();
		 levelsIter != covarDataT.rowUniqLevels[cData].end(); levelsIter++) {
	      if (covarDataT.isMissing(*levelsIter) && !covarUseMissingIndic) continue;
	      cout << "    Using categorical covariate: " << covarDataT.rowNames[cData]
		   << " (adding level " << *levelsIter << ")" << endl;
	      covarIndLevelPairs.push_back(std::make_pair(cData, *levelsIter));
	    }
	    if (covarDataT.rowUniqLevels[cData].size() > 10) {
	      cerr << "WARNING: Covariate " << covarDataT.rowNames[cData]
		   << " has a large number of distinct values ("
		   << covarDataT.rowUniqLevels[cData].size() << ")" << endl;
	      cerr << "         Should this covariate be quantitative rather than categorical?"
		   << endl;
	    }
	    if ((int) covarDataT.rowUniqLevels[cData].size() > covarMaxLevels) {
	      cerr << "ERROR: Number of distinct values of covariate "
		   << covarDataT.rowNames[cData]
		   << " exceeds max (" << covarMaxLevels << ")" << endl;
	      cerr << "       Should this covariate be quantitative rather than categorical?"
		   << endl;
	      cerr << "If not, set --covarMaxLevels to turn off error-check" << endl;
	      exit(1);
	    }
	  }
	  found = true;
	  break;
	}
      if (!found) {
	cerr << "ERROR: Unable to find covariate " << it->first << endl;
	exit(1);
      }
    }

    C = covarIndLevelPairs.size();

    Nused = 0; for (uint64 n = 0; n < Nstride; n++) Nused += (int) _maskIndivs[n];
    int numSamplesMissingCovars = 0;
    for (uint64 n = 0; n < Nstride; n++)
      if (_maskIndivs[n] && !maskIndivs[n])
	numSamplesMissingCovars++;
    if (numSamplesMissingCovars) {
      cerr << "WARNING: " << numSamplesMissingCovars << " of " << Nused
	   << " samples passing previous QC have missing covariates" << endl;
      if (covarUseMissingIndic)
	cerr << "  --covarUseMissingIndic is set, so these samples will still be analyzed" << endl;
      else
	cerr << "  --covarUseMissingIndic is not set, so these samples will be removed" << endl;
    }

    if (covarUseMissingIndic) // overwrite maskIndivs with input mask (use all input maskIndivs)
      memcpy(maskIndivs, _maskIndivs, Nstride * sizeof(maskIndivs[0]));
    else // complete case option: intersect input mask with missing covars mask
      for (uint64 n = 0; n < Nstride; n++)
	maskIndivs[n] *= _maskIndivs[n];

    Nused = 0; for (uint64 n = 0; n < Nstride; n++) Nused += (int) maskIndivs[n];
    cout << "Number of individuals used in analysis: Nused = " << Nused << endl;

    if (C > Nused) {
      cerr << "ERROR: Number of covariates cannot exceed number of individuals" << endl;
      exit(1);
    }
    
    // mask the covariate data matrix with maskIndivs
    // fill in missing covariate values
    // - use covarDataT.missing_key to figure out which are missing; take means over maskIndivs==1
    // - TODO: PLINK seems by default to eliminate indivs with missing covars from analysis?
    double *unmaskedCovars = ALIGNED_MALLOC_DOUBLES(C*Nstride);
    basis = ALIGNED_MALLOC_DOUBLES(C*Nstride);
    for (uint64 c = 0; c < C; c++) {
      uint64 cData = covarIndLevelPairs[c].first;
      const string &covarLevel = covarIndLevelPairs[c].second;
      if (covarLevel == "") { // quantitative
	for (uint64 n = 0; n < Nstride; n++) {
	  double value = covarDataT.getEntryDbl(cData, n);
	  if (value == covarDataT.missing_key_dbl) value = 0; // zero-fill missing values
	  unmaskedCovars[c*Nstride + n] = value;
	}
      }
      else if (covarLevel == QTVE_MISSING_INDICATOR) {
	for (uint64 n = 0; n < Nstride; n++) {
	  double value = covarDataT.getEntryDbl(cData, n);
	  unmaskedCovars[c*Nstride + n] = (value == covarDataT.missing_key_dbl ? 1 : 0);
	}
      }
      else {
	for (uint64 n = 0; n < Nstride; n++)
	  unmaskedCovars[c*Nstride + n] = (covarDataT.data[cData][n] == covarLevel);
      }
      for (uint64 n = 0; n < Nstride; n++) // temporarily holds masked covars; overwritten
	basis[c*Nstride + n] = unmaskedCovars[c*Nstride + n] * maskIndivs[n];
    }

    // compute svd; set Cindep
    double *S = ALIGNED_MALLOC_DOUBLES(C); // singular values
    double *Vtrans = ALIGNED_MALLOC_DOUBLES(C*C); // V' stored in column-major order

    { // A (masked covariates) = U * Sigma * V'
      char JOBU_ = 'O', JOBVT_ = 'A'; // overwrite input matrix with left singular vectors
      int M_ = Nstride, N_ = C;
      double *A_ = basis;
      int LDA_ = Nstride;
      double *S_ = S;
      double *U_ = NULL;
      int LDU_ = 1;
      double *VT_ = Vtrans;
      int LDVT_ = C;
      int LWORK_ = 5*Nstride;
      double *WORK_ = ALIGNED_MALLOC_DOUBLES(LWORK_);
      int INFO_;
      DGESVD_MACRO(&JOBU_, &JOBVT_, &M_, &N_, A_, &LDA_, S_, U_, &LDU_, VT_, &LDVT_,
		   WORK_, &LWORK_, &INFO_);
      ALIGNED_FREE(WORK_);
      if (INFO_ != 0) {
	cerr << "ERROR: SVD computation for covariate matrix failed" << endl;
	exit(1);
      }
    }
    
    cout << "Singular values of covariate matrix:" << endl;
    for (uint64 c = 0; c < C; c++)
      cout << "    S[" << c << "] = " << S[c] << endl;

    for (Cindep = 0; Cindep < C; Cindep++)
      if (S[Cindep] < S[0] * MIN_REL_SINGULAR_VALUE)
	break;
    
    cout << "Total covariate vectors: C = " << C << endl;
    cout << "Total independent covariate vectors: Cindep = " << Cindep << endl;

    basisExtAllIndivs = ALIGNED_MALLOC_DOUBLES(Cindep*Nstride);
    // U ext = A (unmasked covariates) * V * Sigma^-1
    { // A (unmasked covariates) * V
      char TRANSA_ = 'N';
      char TRANSB_ = 'T';
      int M_ = Nstride;
      int N_ = Cindep;
      int K_ = C;
      double ALPHA_ = 1.0;
      double *A_ = unmaskedCovars;
      int LDA_ = Nstride;
      double *B_ = Vtrans;
      int LDB_ = C;
      double BETA_ = 0.0;
      double *C_ = basisExtAllIndivs;
      int LDC_ = Nstride;
      DGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		  &BETA_, C_, &LDC_);
    }
    // multiply by Sigma^-1
    for (uint64 c = 0; c < Cindep; c++) {
      double invS = 1.0/S[c];
      for (uint64 n = 0; n < Nstride; n++)
	basisExtAllIndivs[c*Nstride+n] *= invS;
    }
    /*
    for (uint64 c = 0; c < Cindep; c++)
      for (uint64 n = 0; n < Nstride; n++)
	if (fabs(basisExtAllIndivs[c*Nstride+n] - basis[c*Nstride+n]) > 1e-6) {
	  cout << "basis c=" << c << " n=" << n << ": " << basis[c*Nstride+n] << " ext: " << basisExtAllIndivs[c*Nstride+n] << endl;
	}
    */
    ALIGNED_FREE(Vtrans);
    ALIGNED_FREE(S);
    ALIGNED_FREE(unmaskedCovars);
  }

  CovariateBasis::~CovariateBasis() {
    ALIGNED_FREE(maskIndivs);
    ALIGNED_FREE(basis);
    ALIGNED_FREE(basisExtAllIndivs);
  }

  //uint64 getC(void) const { return C; } -- don't expose this!
  uint64 CovariateBasis::getCindep(void) const { return Cindep; }
  const double *CovariateBasis::getMaskIndivs(void) const { return maskIndivs; }
  uint64 CovariateBasis::getNused(void) const { return Nused; }
  const double *CovariateBasis::getBasis(bool extAllIndivs) const {
    return extAllIndivs ? basisExtAllIndivs : basis;
  }
  
  // vec is assumed to have dimension Nstride
  // don't assume memory alignment (no SSE)
  void CovariateBasis::applyMaskIndivs(double vec[]) const {
    for (uint64 n = 0; n < Nstride; n++)
      vec[n] *= maskIndivs[n];
  }

  /**
   * Cindep values will be written to out[]
   * vec[] has size Nstride
   */
  void CovariateBasis::computeCindepComponents(double out[], const double vec[]) const {
    char TRANS = 'T';
    int M = Nstride, N = Cindep;
    double ALPHA = 1;
    // A = basis
    int LDA = Nstride;
    // X = vec
    int INCX = 1;
    double BETA = 0;
    // Y = out
    int INCY = 1;
    DGEMV_MACRO(&TRANS, &M, &N, &ALPHA, basis, &LDA, vec, &INCX, &BETA, out, &INCY);
  }

  /**
   * vec[] has size Nstride
   * don't assume memory alignment (no SSE)
   */
  void CovariateBasis::projectCovars(double vec[]) const {
    double *covarComps = ALIGNED_MALLOC_DOUBLES(Cindep);
    double *vecAligned = ALIGNED_MALLOC_DOUBLES(Nstride);
    memcpy(vecAligned, vec, Nstride*sizeof(vec[0]));
    computeCindepComponents(covarComps, vecAligned);
    for (uint64 c = 0; c < Cindep; c++)
      for (uint64 n = 0; n < Nstride; n++)
	vec[n] -= covarComps[c] * basis[c*Nstride + n];
    ALIGNED_FREE(vecAligned);
    ALIGNED_FREE(covarComps);
  }

  // debugging function
  void CovariateBasis::printProj(const double vec[], const char name[]) const {
    double copyVec[Nstride]; memcpy(copyVec, vec, Nstride*sizeof(vec[0]));
    projectCovars(copyVec);
    printf("%s:\n", name);
    for (uint64 n = 0; n < Nstride && n < 20; n++)
      printf("%f\n", copyVec[n]);
    printf("\n");
  }
}
