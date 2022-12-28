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

#ifndef COVARIATEBASIS_HPP
#define COVARIATEBASIS_HPP

#include <vector>
#include <string>
#include <utility>
#include <boost/utility.hpp>

#include "DataMatrix.hpp"

namespace LMM {

  class CovariateBasis : boost::noncopyable {

  private:
    uint64 C; // number of covariates (including all-1s) chosen from DataMatrix of covariates
    uint64 Cindep; // number of independent covariates
    uint64 Nstride; // minor dimension of maskedCovars and basis arrays (for data layout)
    uint64 Nused; // popcnt(maskIndivs)

    // beyond input _maskIndivs, samples with missing covariates are masked if !covarUseMissingIndic
    double *maskIndivs; // [VECTOR]: 0-1 vector of length Nstride
    double *basis; // [[MATRIX]]: C x Nstride; normalized to have vector norm 1 (unlike snps)
    double *basisExtAllIndivs; // [[MATRIX]]: Cindep x Nstride; basis "extended" for OOS prediction

  public:
    static const double MIN_REL_SINGULAR_VALUE;

    CovariateBasis(const DataMatrix &covarDataT, const double _maskIndivs[],
		   const std::vector < std::pair <std::string, DataMatrix::ValueType> > &covars,
		   int covarMaxLevels, bool covarUseMissingIndic);

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
    void init(const DataMatrix &covarDataT, const double _maskIndivs[],
	      std::vector < std::pair <std::string, DataMatrix::ValueType> > covars,
	      int covarMaxLevels, bool covarUseMissingIndic);

    ~CovariateBasis();

    //uint64 getC(void) const { return C; } -- don't expose this!
    uint64 getCindep(void) const;
    const double *getMaskIndivs(void) const;
    uint64 getNused(void) const;
    const double *getBasis(bool extAllIndivs) const;
  
    // vec is assumed to have dimension Nstride
    // don't assume memory alignment (no SSE)
    void applyMaskIndivs(double vec[]) const;

    /**
     * Cindep values will be written to out[]
     * vec[] has size Nstride
     */
    void computeCindepComponents(double out[], const double vec[]) const;

    /**
     * vec[] has size Nstride
     * don't assume memory alignment (no SSE)
     */
    void projectCovars(double vec[]) const;

    // debugging function
    void printProj(const double vec[], const char name[]) const;

  };
}

#endif
