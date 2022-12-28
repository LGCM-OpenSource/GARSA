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

#ifndef DATAMATRIX_HPP
#define DATAMATRIX_HPP

#include <vector>
#include <string>
#include <set>

#include <boost/utility.hpp>

#include "SnpData.hpp"

namespace LMM {

  class DataMatrix : boost::noncopyable {
  
    static const double DEFAULT_MISSING_KEY_DBL;
    static const std::string DEFAULT_MISSING_KEY_STR;

  public:
    static const std::string ALL_ONES_NAME;

    enum ValueType {
      QUANTITATIVE,
      CATEGORICAL
    };

    uint64 nrows, ncols;

    // [VECTOR]: covariates are internally stored as rows
    std::vector <std::string> rowNames;

    // [VECTOR]: unique values (including missing) among row entries; used for categorical covars
    std::vector < std::set <std::string> > rowUniqLevels;

    // [[MATRIX]]: nrows x ncols (transposed from input format!)
    std::vector <std::vector <std::string> > data;

    std::string missing_key_str;
    double missing_key_dbl;

    DataMatrix(uint64 _nrows, uint64 _ncols);

    // match (FID, IID) to snpData; output number of indivs matched
    // create C x Nstride matrix (transposed from input format, which is NcovarFile x C matrix!!)
    DataMatrix(const std::string &covarFile, const SnpData &snpData,
	       const std::vector <std::string> &colNames);

    // reads data from file and transposes it; 1st row = column headers, stored into rowNames
    DataMatrix(const std::string &file, std::string _missing_key_str=DEFAULT_MISSING_KEY_STR,
	       double _missing_key_dbl=DEFAULT_MISSING_KEY_DBL);

    double parseDouble(const std::string &strValue) const;
    double getEntryDbl(uint64 r, uint64 c) const;
    uint64 getRowIndex(const std::string &rowName) const;
    // error if rowName is not present
    std::vector <std::string> getRowStr(const std::string &rowName) const;
    // error if rowName is not present
    std::vector <double> getRowDbl(const std::string &rowName) const;

    std::vector <double> lookupKeyValuesDbl(const std::vector <std::string> &keys,
					    const std::string &keyName,
					    const std::string &valueName) const;
    bool isMissing(const std::string &str) const;
  };
}

#endif
