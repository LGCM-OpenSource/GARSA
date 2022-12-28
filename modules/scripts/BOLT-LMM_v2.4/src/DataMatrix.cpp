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

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <set>

#include "FileUtils.hpp"
#include "SnpData.hpp"
#include "DataMatrix.hpp"

namespace LMM {

  using std::vector;
  using std::string;
  using std::cout;
  using std::cerr;
  using std::endl;
  using FileUtils::getline;

  const double DataMatrix::DEFAULT_MISSING_KEY_DBL = -9;
  const string DataMatrix::DEFAULT_MISSING_KEY_STR = "NA";
  const string DataMatrix::ALL_ONES_NAME = "CONST_ALL_ONES";

  DataMatrix::DataMatrix(uint64 _nrows, uint64 _ncols) : nrows(_nrows), ncols(_ncols) {
    data = vector < vector <string> > (nrows, vector <string> (ncols));
  }

  // assumes only quantitative covars; for categorical, use plink --write-covar --dummy-coding
  // match (FID, IID) to snpData; output number of indivs matched
  // create C x Nstride matrix (transposed from input format, which is NcovarFile x C matrix!!)
  DataMatrix::DataMatrix(const string &covarFile, const SnpData &snpData,
			 const vector <string> &colNames) {
    // header begins: FID IID...

    missing_key_dbl = DEFAULT_MISSING_KEY_DBL;
    missing_key_str = DEFAULT_MISSING_KEY_STR;

    std::set <string> colNamesSet(colNames.begin(), colNames.end());
    vector <bool> inNames; // record whether to keep each column (= row in transposed version)
    rowNames = vector <string> (1, ALL_ONES_NAME); // always include all-1s vector
    FileUtils::AutoGzIfstream fin;

    // find number of covariates (nrows) and store names
    if (!covarFile.empty()) { // process header line
      fin.openOrExit(covarFile);
      string line;
      getline(fin, line);
      std::istringstream iss(line);
      string FID, IID, covarName;
      iss >> FID >> IID;
      if (FID != "FID" || IID != "IID") {
	cerr << "ERROR: Phenotype/covariate file must start with header: FID IID" << endl;
	cerr << "       " << covarFile << endl;
	exit(1);
      }
      while (iss >> covarName) {
	inNames.push_back(colNamesSet.count(covarName));
	if (inNames.back())
	  rowNames.push_back(covarName);
      }
    }

    // allocate memory and initialize first covariate to all-1s vector
    int nPerLine = inNames.size(); // number of columns in file after FID IID
    nrows = rowNames.size(); // note this includes all-1s vector
    ncols = snpData.getNstride();
    // NOTE: columns of data[] correspond to indivs
    // first covariate is all-1s vector; initialize all remaining entries to missing
    data = vector < vector <string> > (nrows, vector <string> (ncols, missing_key_str));
    data[0] = vector <string> (ncols, "1"); 

    std::set <uint64> indivsSeen;

    // input covariate data into data matrix
    if (!covarFile.empty()) {
      string line, FID, IID;
      int numLines = 0, numIgnore = 0;
      while (getline(fin, line)) {
	numLines++;
	std::istringstream iss(line);
	iss >> FID >> IID;
	uint64 n = snpData.getIndivInd(FID, IID);
	if (n == SnpData::IND_MISSING) {
	  if (numIgnore < 5)
	    cerr << "WARNING: Ignoring indiv not in genotype data: FID=" << FID << ", IID=" << IID
		 << endl;
	  numIgnore++;
	}
	else {
	  if (indivsSeen.count(n)) {
	    cerr << "WARNING: Duplicate entry for indiv FID=" << FID << ", IID=" << IID << endl;
	  }
	  else indivsSeen.insert(n);

	  string covarValue;
	  vector <string> covars;
	  int ctr = 0;
	  while (iss >> covarValue) {
	    if (inNames[ctr]) // skip columns not requested
	      covars.push_back(covarValue);
	    ctr++;
	  }
	  if (ctr != nPerLine) {
	    cerr << "ERROR: Wrong number of entries in data row:" << endl;
	    cerr << line << endl;
	    cerr << "Expected " << nPerLine << " fields after FID, IID cols" << endl;
	    cerr << "Parsed " << covars.size() << " fields" << endl;
	    exit(1);
	  }
	  for (uint64 iCovar = 0; iCovar < covars.size(); iCovar++)
	    data[iCovar+1][n] = covars[iCovar]; // offset by 1 because of all-1s vector
	}
      }
      fin.close();
#ifdef VERBOSE
      cout << "Read data for " << numLines << " indivs (ignored " << numIgnore
	   << " without genotypes) from:\n  " << covarFile << endl;
#endif
    }

    // save unique values in each row
    rowUniqLevels = vector < std::set <string> > (nrows);
    for (uint64 r = 0; r < nrows; r++) // don't include dummy indivs (set to missing) if Nstride>N
      rowUniqLevels[r] = std::set <string> (data[r].begin(),
					    data[r].begin() + snpData.getNumIndivsQC());
  }

  // reads data from file and transposes it; 1st row = column headers, stored into rowNames
  DataMatrix::DataMatrix(const string &file, string _missing_key_str, double _missing_key_dbl) :
    missing_key_str(_missing_key_str), missing_key_dbl(_missing_key_dbl) {
    
    FileUtils::AutoGzIfstream fin; fin.openOrExit(file);
    string line;

    // process header line: store names of columns (transposed to rows)
    getline(fin, line);
    std::istringstream issHead(line);
    string headerName;
    while (issHead >> headerName) rowNames.push_back(headerName);
    nrows = rowNames.size();

    // count number of remaining lines: data rows (transposed to cols)
    ncols = 0;
    while (getline(fin, line))
      ncols++;
    
    // initialize data matrix
    data = vector < vector <string> > (nrows, vector <string> (ncols));

    fin.clear(); fin.seekg(0, std::ios::beg); // rewind
    getline(fin, line); // throw out header

    // read and store data
    for (uint64 c = 0; c < ncols; c++) {
      getline(fin, line);
      std::istringstream iss(line);
      string value;
      vector <string> values;
      while (iss >> value) values.push_back(value);
      if (values.size() != nrows) {
	cerr << "ERROR: Wrong number of entries in data row:" << endl;
	cerr << line << endl;
	cerr << "Expected " << nrows << " fields; parsed " << values.size() << endl;
	exit(1);
      }
      for (uint64 r = 0; r < nrows; r++)
	data[r][c] = values[r];
    }
    fin.close();
    
    // todo: save unique values in each row?
  }

  double DataMatrix::parseDouble(const string &strValue) const {
    if (strValue == missing_key_str)
      return missing_key_dbl;
    else {
      double d;
      int success = sscanf(strValue.c_str(), "%lf", &d);
      if (success)
	return d;
      else {
	cerr << "ERROR: Could not parse DataMatrix field to floating-point: " << strValue << endl;
	exit(1);
      }
    }
  }

  double DataMatrix::getEntryDbl(uint64 r, uint64 c) const {
    return parseDouble(data[r][c]);
  }

  uint64 DataMatrix::getRowIndex(const std::string &rowName) const {
    for (uint64 r = 0; r < nrows; r++)
      if (rowNames[r] == rowName)
	return r;
    cerr << "ERROR: Unable to find field named " << rowName << endl;
    exit(1);
  }

  // error if rowName is not present
  vector <string> DataMatrix::getRowStr(const string &rowName) const {
    uint64 r = getRowIndex(rowName);
    vector <string> rowDataStr(ncols);
    for (uint64 i = 0; i < ncols; i++)
      rowDataStr[i] = data[r][i];
    return rowDataStr;
  }

  // error if rowName is not present
  vector <double> DataMatrix::getRowDbl(const string &rowName) const {
    vector <string> rowDataStr = getRowStr(rowName);
    vector <double> rowDataDbl(ncols);
    for (uint64 i = 0; i < ncols; i++)
      rowDataDbl[i] = parseDouble(rowDataStr[i]);
    return rowDataDbl;
  }

  /*
  void DataMatrix::writeMatrixDbl(double dest[]) const {
    for (uint64 r = 0; r < nrows; r++)
      for (uint64 i = 0; i < ncols; i++)
	dest[r*ncols + i] = parseDouble(data[r][i]);
  }
  */

  vector <double> DataMatrix::lookupKeyValuesDbl(const vector <string> &keys,
						 const string &keyName, const string &valueName)
    const {
    
    uint64 rKey = getRowIndex(keyName);
    uint64 rValue = getRowIndex(valueName);
    std::map <string, double> lut;
    for (uint64 c = 0; c < ncols; c++) {
      if (lut.find(data[rKey][c]) != lut.end())
	cerr << "WARNING: Duplicate key " << data[rKey][c] << " -- ignoring duplicate" << endl;
      else
	lut[data[rKey][c]] = parseDouble(data[rValue][c]);
    }
    vector <double> values(keys.size(), missing_key_dbl);
    int found = 0;
    for (uint64 i = 0; i < keys.size(); i++)
      if (lut.find(keys[i]) != lut.end()) {
	values[i] = lut[keys[i]];
	found++;
      }
#ifdef VERBOSE
    cout << "Found " << found << "/" << keys.size() << " keys" << endl;
#endif
    return values;
  }

  bool DataMatrix::isMissing(const string &str) const {
    double d; int success = sscanf(str.c_str(), "%lf", &d);
    return str == missing_key_str || (success && d == missing_key_dbl);
  }

};
