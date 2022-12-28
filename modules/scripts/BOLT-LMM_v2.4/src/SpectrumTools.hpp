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

#ifndef SPECTRUMTOOLS_HPP
#define SPECTRUMTOOLS_HPP

#include <map>
#include <vector>

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

  std::map <double, double> tallySpectrum(const std::vector <double> &MAFs,
					  const std::vector <double> UBs);

  void writeSpectrum(const std::string &fileName, const std::map <double, double> &spectrum);

  std::map <double, double> readSpectrum(const std::string &fileName);
  
  std::vector <double> computeFreqRatios(const std::vector <double> &MAFs,
					 const std::map <double, double> &refSpectrum);
}

#endif
