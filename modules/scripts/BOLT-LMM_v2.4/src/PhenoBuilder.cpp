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

#include <cstdlib>
#include <vector>
#include <string>
#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>

#include "Types.hpp"
#include "FileUtils.hpp"
#include "NumericUtils.hpp"
#include "SpectrumTools.hpp"
#include "Bolt.hpp"
#include "PhenoBuilder.hpp"

namespace LMM {

  using std::vector;
  using std::string;
  using std::set;
  using std::cout;
  using std::cerr;
  using std::endl;

  // build masked, mean-centered, normalized genetic component of phenotype from input betas
  vector <double> PhenoBuilder::buildPhenoGenetic(const vector <double> &betas) {
    vector <double> pheno(Nstride);
    bolt.batchComputePreds(&pheno[0], &betas[0], NULL, 1, false);
    maskCenterNormalize(pheno); // mask, mean-center, normalize (first two already done)
    return pheno;
  }

  bool PhenoBuilder::isNullRegion(PhenoBuilder::SnpRegionType snpRegion) {
    return snpRegion == PhenoBuilder::NULL_REGION_H2HI_CHROMS
      || snpRegion == PhenoBuilder::NULL_REGION_H2LO_CHROMS;
  }

  /**
   * input: pheno is assumed to have dimension Nstride
   * - applies maskIndivs
   * - mean-centers (using mean on remaining indivs!)
   * - normalizes so that mean sq entry is 1 -- unless all 0s
   */
  void PhenoBuilder::maskCenterNormalize(vector <double> &pheno) {
    bolt.getCovBasis().applyMaskIndivs(&pheno[0]);
    uint64 Nused = bolt.getCovBasis().getNused();
    double mean = NumericUtils::mean(&pheno[0], Nstride, Nused);      
    for (uint64 n = 0; n < Nstride; n++)
      pheno[n] -= mean;
    // re-zero after subtracting mean
    bolt.getCovBasis().applyMaskIndivs(&pheno[0]);
    double norm2 = NumericUtils::norm2(&pheno[0], Nstride);
    if (norm2 > 0) {
      double scale = sqrt(Nused / norm2);
      for (uint64 n = 0; n < Nstride; n++)
	pheno[n] *= scale;
    }
  }

  const int PhenoBuilder::DEFAULT_MID_CHROM_HALF_BUFFER_PHYSPOS = 2000000; // +/- 2 Mb
  const double PhenoBuilder::DEFAULT_MID_CHROM_HALF_BUFFER_GENPOS = 0.02; // +/- 2 cM
    
  // note: the Bolt instance should apply any additional masking needed (indivs, snps)
  PhenoBuilder::PhenoBuilder(const Bolt &_bolt, uint seed, EffectDistType _effectDist) :
    bolt(_bolt),
    effectDist(_effectDist),
    rng(seed+123456789),
    randn(rng, boost::normal_distribution<>(0.0, 1.0)),
    rand(rng, boost::uniform_01<>()),
    exprnd(rng, boost::exponential_distribution<>()) {

    Nstride = bolt.getSnpData().getNstride();
  }

  /**
   * build masked, mean-centered, normalized causal snp genetic component of phenotype
   *
   * params:
   * - pCausal: prob of a 1st-half chrom being causal, BEFORE further reduction from MAF matching
   * - MAFhistFile: file containing allele freq spectrum to match
   * - stdPow: MAF-dependence; 0 for equal per-normalized-genotype, 1 for equal per-allele
  */
  vector <double> PhenoBuilder::genPhenoCausal(vector <double> &simBetas,
					       vector <SnpRegionType> &simRegions, int vcNum,
					       int Mcausal, const string &MAFhistFile,
					       double stdPow, set <int> highH2Chroms,
					       int midChromHalfBufferPhyspos) {

    const vector <SnpInfo> &snps = bolt.getSnpData().getSnpInfo();
    uint64 M = snps.size();
    const uchar *projMaskSnps = bolt.getProjMaskSnps();

    simBetas = vector <double> (M);
    simRegions = vector <SnpRegionType> (M);

    /*
     * spectrum adjustment:
     * - make similar hist of MAFs in GWAS data set
     * - boostRatio(bin) := (ref count / GWAS count)
     * - multiply in p_causal(bin) factor of boostRatio(bin) / max(boostRatios)
     */
    vector <double> freqRatios(M, 1.0);
    if (!MAFhistFile.empty()) {
      vector <double> MAFs(M);
      for (uint64 m = 0; m < M; m++) MAFs[m] = snps[m].MAF;;
      freqRatios =
	SpectrumTools::computeFreqRatios(MAFs, SpectrumTools::readSpectrum(MAFhistFile));
    }

    // find middle snp in each chrom
    std::map <int, uint64> firstSnpIndex, lastSnpIndex, midSnpIndex;
    for (uint64 m = 0; m < M; m++) {
      if (firstSnpIndex.find(snps[m].chrom) == firstSnpIndex.end())
	firstSnpIndex[snps[m].chrom] = m;
      lastSnpIndex[snps[m].chrom] = m;
    }
    for (std::map <int, uint64>::iterator it = firstSnpIndex.begin(); it != firstSnpIndex.end();
	 it++) {
      int chrom = it->first;
      midSnpIndex[chrom] = (firstSnpIndex[chrom] + lastSnpIndex[chrom]) / 2;
    }
    // set midChromHalfBufferGenpos (0 if no genetic map)
    double midChromHalfBufferGenpos = (bolt.getSnpData().getMapAvailable() ?
				       DEFAULT_MID_CHROM_HALF_BUFFER_GENPOS : 0);
    
    uint64 numSnps = 0;
    vector <int> causalRegionSnps;
    for (uint64 m = 0; m < M; m++) {
      if (projMaskSnps[m] && snps[m].vcNum == vcNum) {
	numSnps++;

	// set simRegions
	uint64 m_mid = midSnpIndex[snps[m].chrom];
	if (snps[m].physpos <= snps[m_mid].physpos - midChromHalfBufferPhyspos &&
	    snps[m].genpos <= snps[m_mid].genpos - midChromHalfBufferGenpos)
	  simRegions[m] = CAUSAL_REGION; // in first half, before buffer region
	else if (snps[m].physpos >= snps[m_mid].physpos + midChromHalfBufferPhyspos &&
		 snps[m].genpos >= snps[m_mid].genpos + midChromHalfBufferGenpos) {
	  // in second half, after buffer region
	  if (highH2Chroms.count(snps[m].chrom))
	    simRegions[m] = NULL_REGION_H2HI_CHROMS;
	  else
	    simRegions[m] = NULL_REGION_H2LO_CHROMS;
	}
	else // in buffer region
	  simRegions[m] = BUFFER_REGION;

	// only first-half snps on highH2Chroms can have effects...
	if ((highH2Chroms.count(snps[m].chrom) && m < m_mid && freqRatios[m] > 1e-4)
	    || midChromHalfBufferPhyspos == -1) // ... unless flag is set to turn off 1st/2nd half
	  causalRegionSnps.push_back(m);
      }
      else // flagged to ignore in bolt
	simRegions[m] = BAD_REGION;
    }

    int numCausalRegionSnps = causalRegionSnps.size();
    if (numCausalRegionSnps < Mcausal) {
      cerr << "ERROR: Not enough SNPs in causal regions to generate " << Mcausal << " causal SNPs"
	   << endl;
      exit(1);
    }

    for (int t = 0; t < Mcausal; t++)
      while (true) {
	int i = (int) (rand() * causalRegionSnps.size());
	uint64 m = causalRegionSnps[i];
	if (rand() < freqRatios[m]) { // downsample to match MAF hist from MAFhistFile
	  // sample beta with scale according to stdPow
	  double effectScale = pow(snps[m].MAF * (1-snps[m].MAF), 0.5*stdPow);
	  simBetas[m] = effectScale *
	    (effectDist == GAUSSIAN ? randn() : (exprnd() * (rand()<0.5 ? -1 : 1)));
	  std::swap(causalRegionSnps[i], causalRegionSnps.back());
	  causalRegionSnps.pop_back();
	  break;
	}
      }
    
#ifdef VERBOSE
    cout << "Number of non-masked SNPs (including second halves of chroms): " << numSnps << endl;
    cout << "Generated " << Mcausal << " causal SNP effects from "
	 << numCausalRegionSnps << " available SNPs" << endl;
#endif

    return buildPhenoGenetic(simBetas);
  }

  vector <double> PhenoBuilder::genPhenoCausalRegions(vector <double> &simBetas,
						      vector <SnpRegionType> &simRegions,
						      double lambdaRegion, double pRegion) {

    const vector <SnpInfo> &snps = bolt.getSnpData().getSnpInfo();
    uint64 M = snps.size();
    const uchar *projMaskSnps = bolt.getProjMaskSnps();

    simBetas = vector <double> (M);
    simRegions = vector <SnpRegionType> (M, CAUSAL_REGION);

    vector <uint64> mRegionEdges(1, 0);
    while (mRegionEdges.back() != M) {
      uint64 mLast = mRegionEdges.back();
      //cout << mLast << "\t" << snps[mLast].chrom << "\t" << snps[mLast].physpos << endl;
      double chrBp = 1e9*snps[mLast].chrom + snps[mLast].physpos + lambdaRegion*1e6*exprnd();
      uint64 mNext = mLast+1;
      while (mNext < M && 1e9*snps[mNext].chrom + snps[mNext].physpos < chrBp)
	mNext++;
      mRegionEdges.push_back(mNext);
    }
    cout << "Generated " << mRegionEdges.size()-1 << " regions" << endl;
    int hotRegions = 0, hotSnps = 0;
    vector <double> hotLengths;
    for (uint i = 0; i+1 < mRegionEdges.size(); i++)
      if (rand() <= pRegion) {
	hotRegions++;
	hotLengths.push_back(1e-6 * (snps[mRegionEdges[i+1]-1].physpos
				     - snps[mRegionEdges[i]].physpos));
	for (uint64 m = mRegionEdges[i]; m < mRegionEdges[i+1]; m++)
	  if (projMaskSnps[m]) {
	    hotSnps++;
	    simBetas[m] = randn();
	  }
      }    
    cout << "Generated " << hotSnps << " causal SNP effects in " << hotRegions << " regions"
	 << endl;
    printf("Hot region length min = %.3f, max = %.3f, mean = %.3f Mb\n",
	   *std::min_element(hotLengths.begin(), hotLengths.end()),
	   *std::max_element(hotLengths.begin(), hotLengths.end()),
	   std::accumulate(hotLengths.begin(), hotLengths.end(), 0.0) / hotLengths.size());
    return buildPhenoGenetic(simBetas);
  }

  /**
   * build masked, mean-centered, normalized candidate snp genetic component of phenotype
   * update simRegions accordingly to identify chosen candidate snps
   */
  vector <double> PhenoBuilder::genPhenoCandidate(vector <SnpRegionType> &simRegions,
						  int Mcandidate) {
    uint64 M = simRegions.size();
    int numCausalRegion = 0;
    for (uint64 m = 0; m < M; m++) if (simRegions[m] == CAUSAL_REGION) numCausalRegion++;
    if (numCausalRegion < Mcandidate) {
      cerr << "ERROR: Too few SNPs in causal region: " << numCausalRegion
	   << " < Mcandidate=" << Mcandidate << endl;
      exit(1);
    }
    vector <double> simBetas = vector <double> (M);
    for (int t = 0; t < Mcandidate; t++)
      while (true) {
	uint64 m = (uint64) (rand() * M);
	if (simRegions[m] == CAUSAL_REGION) {
	  simRegions[m] = CANDIDATE_SNP;
	  simBetas[m] = 1;
	  break;
	}
      }
#ifdef VERBOSE
    cout << "Generated " << Mcandidate << " candidate SNP effects" << endl;
#endif
    return buildPhenoGenetic(simBetas);
  }

  // build masked, mean-centered, normalized stratification component of phenotype
  vector <double> PhenoBuilder::genPhenoStrat(const string &phenoStratFile) {
    vector <double> pheno(Nstride);    
    // todo (clean up later)
    // for now, just assume the file contains a pheno stratification vector
    FileUtils::AutoGzIfstream finPhenoStrat; finPhenoStrat.openOrExit(phenoStratFile);
    uint64 ctr = 0;
    while (finPhenoStrat >> pheno[ctr]) {
      ctr++;
      if (ctr >= Nstride) { // note Bolt instance doesn't know the real N; just don't seg fault 
	cerr << "ERROR: Too many records in phenoStratFile: " << phenoStratFile << endl;
	cerr << "       At most N (# of indivs) allowed" << endl;
	exit(1);
      }
    }
    cout << "Read " << ctr << " entries of pheno stratification vector" << endl;
    maskCenterNormalize(pheno); return pheno;
  }

  /**
   * build masked, mean-centered, normalized relatedness component of phenotype
   * - input matrix will be N x N
   * - also input ibd thresh
   * - mask (i,j) entries for i!=j, at least one of i, j in masked-out indivs
   * - cholesky factor
   * - create random pheno
   * - mask, mean-center, normalize
   // input file: IBS mat (unmasked: N x N for N indivs in fam file... need to augment if < Nstride)
   // compute IBS mat with GCTA; only do this for FHS testing on relatedness
   vector <double> genPhenoRelated(const string &IBSfile, double IBSthresh) {
   vector <double> pheno(Nstride);
   // todo (later)
   maskCenterNormalize(pheno); return pheno;
   }
  */

  // build masked, mean-centered, normalized environmental (random) component of phenotype
  vector <double> PhenoBuilder::genPhenoEnv(void) {
    vector <double> pheno(Nstride);
    for (uint64 n = 0; n < Nstride; n++) pheno[n] = randn();
    maskCenterNormalize(pheno); return pheno;
  }

  vector <double> PhenoBuilder::combinePhenoComps
  (const vector <double> &h2causal, double h2candidate, double h2strat,
   const vector < vector <double> > &phenoCausal, const vector <double> &phenoCandidate,
   const vector <double> &phenoStrat, const vector <double> &phenoEnv) {
    vector <double> pheno(Nstride);
    // ignore any components with 0 weight (don't ever use the vector)
    for (uint v = 0; v < h2causal.size(); v++)
      if (h2causal[v] != 0)
	for (uint64 n = 0; n < Nstride; n++) pheno[n] += sqrt(h2causal[v])*phenoCausal[v][n];
    if (h2candidate != 0)
      for (uint64 n = 0; n < Nstride; n++) pheno[n] += sqrt(h2candidate)*phenoCandidate[n];
    if (h2strat != 0)
      for (uint64 n = 0; n < Nstride; n++) pheno[n] += sqrt(h2strat)*phenoStrat[n];
    double h2left = 1-std::accumulate(h2causal.begin(), h2causal.end(), 0.0)-h2candidate-h2strat;
    if (h2left > 0)
      for (uint64 n = 0; n < Nstride; n++) pheno[n] += sqrt(h2left)*phenoEnv[n];
    return pheno;
  }
}
