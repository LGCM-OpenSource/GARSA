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

#include <map>
#include <vector>
#include <algorithm>

#include "Types.hpp"
#include "FileUtils.hpp"
#include "SpectrumTools.hpp"

/**
 * functionality:
 * 1) given vector <double> of MAFs (e.g., 1KG eur.maf col) and vector <double> of bin upper bounds
 *    - build a table (map) of frequencies for each bin
 *    - output to file
 *    ex: data/ALL_1000G_phase1integrated_v3_impute/ALL_1000G_phase1integrated_v3_chr%d_impute.gz
 *        create a table 0.01 #SNPs<=0.01 // 0.02 #SNPs btwn (0.01..0.02] // ...
 *
 * 2) given vector <double> of MAFs (e.g., potential causal SNPs for simulation) and ref spectrum
 *    - input reference MAF spectrum from file
 *    - build a table (map) of frequencies (in input SNPs) for each bin from file
 *    - build a table (map) of frequency multipliers for each bin to match the reference spectrum
 *    - return a vector <double> containing the multiplier for each input SNP
 *
 * usage:
 * 1) vector <double> MAFs = [read MAFs from reference data set]
 *    vector <double> UBs = [specify bin upper bounds]
 *    writeSpectrum(fileName, tallySpectrum(MAFs, UBs));
 * 2) vector <double> MAFs = [create MAF vector from SnpData]
 *    vector <double> freqRatios = computeFreqRatios(MAFs, readSpectrum(fileName));
 *    [use in simulation; adjust by global multiplier to match pCausal as well as possible?]
 */
namespace SpectrumTools {

  using std::string;
  using std::vector;
  using std::map;
  using std::endl;

  map <double, double> tallySpectrum(const vector <double> &MAFs, const vector <double> UBs) {
    map <double, double> spectrum;
    for (uint64 i = 0; i < UBs.size(); i++)
      spectrum[UBs[i]] = 0;
    const double snpContrib = 1.0 / MAFs.size();
    for (uint64 m = 0; m < MAFs.size(); m++)
      spectrum.lower_bound(MAFs[m])->second += snpContrib;
    return spectrum;
  }

  void writeSpectrum(const string &fileName, const map <double, double> &spectrum) {
    FileUtils::AutoGzOfstream fout; fout.openOrExit(fileName);
    for (__typeof(spectrum.begin()) it = spectrum.begin(); it != spectrum.end(); it++)
      fout << it->first << '\t' << it->second << endl;
    fout.close();
  }

  map <double, double> readSpectrum(const string &fileName) {
    map <double, double> spectrum;
    FileUtils::AutoGzIfstream fin; fin.openOrExit(fileName);
    double MAF, UB;
    while (fin >> MAF >> UB)
      spectrum[MAF] = UB;
    fin.close();
    return spectrum;
  }
  
  vector <double> computeFreqRatios(const vector <double> &MAFs,
				    const map <double, double> &refSpectrum) {
    vector <double> UBs;
    for (__typeof(refSpectrum.begin()) it = refSpectrum.begin(); it != refSpectrum.end(); it++)
      UBs.push_back(it->second);
    map <double, double> spectrum = tallySpectrum(MAFs, UBs);
    vector <double> freqRatios(MAFs.size());
    double maxMultiplier = 0;
    for (uint64 m = 0; m < MAFs.size(); m++) {
      freqRatios[m] =
	refSpectrum.lower_bound(MAFs[m])->second / spectrum.lower_bound(MAFs[m])->second;
      if (MAFs[m] != 0 && freqRatios[m] > maxMultiplier)
	maxMultiplier = freqRatios[m];
    }
    for (uint64 m = 0; m < MAFs.size(); m++)
      freqRatios[m] /= maxMultiplier;
    return freqRatios;
  }
}
