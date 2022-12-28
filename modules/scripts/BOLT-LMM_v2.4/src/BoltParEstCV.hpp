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

#ifndef BOLTPARESTCV_HPP
#define BOLTPARESTCV_HPP

#include <vector>
#include <string>
#include <utility>
#include <boost/utility.hpp>

#include "Bolt.hpp"
#include "DataMatrix.hpp"
#include "SnpData.hpp"

namespace LMM {

  class BoltParEstCV : boost::noncopyable {

  private:
    const SnpData &snpData;
    const DataMatrix &covarDataT; // transposed covariate data matrix
    const Bolt bolt; // analyses will be performed on indivs in bolt.getMaskIndivs()
    const std::vector < std::pair <std::string, DataMatrix::ValueType> > covars;
    const bool covarUseMissingIndic;

    struct ParamData {
      double f2;
      double p;
      std::vector <double> PVEs, MSEs;
      ParamData(double _f2, double _p);
      bool operator < (const ParamData &paramData2) const;
    };

  public:

    BoltParEstCV(const SnpData& _snpData, const DataMatrix& _covarDataT,
		 const double subMaskIndivs[],
		 const std::vector < std::pair <std::string, DataMatrix::ValueType> > &_covars,
		 int covarMaxLevels, bool missingIndicator, int mBlockMultX, int Nautosomes);
    
    /**
     * (f2, p) parameter estimation via cross-validation
     * - after each fold, compare PVEs of putative (f2, p) param pairs
     * - eliminate clearly suboptimal param pairs from future folds
     * - stop when only one param pair left
     *
     * return: iterations used in last CV fold
     */
    int estMixtureParams
    (double *f2Est, double *pEst, double *predBoost, const std::vector <double> &pheno, 
     double logDeltaEst, double sigma2Kest, int CVfoldsSplit, int CVfoldsCompute,
     bool CVnoEarlyExit, double predBoostMin, bool MCMC, int maxIters, double approxLLtol,
     int mBlockMultX, int Nautosomes) const;

    // for use in PhenoBuilder to generate random phenotypes
    const Bolt &getBoltRef(void) const;

  };
}

#endif
