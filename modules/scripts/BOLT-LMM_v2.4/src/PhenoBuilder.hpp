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

#ifndef PHENOBUILDER_HPP
#define PHENOBUILDER_HPP

#include <vector>
#include <string>
#include <set>
#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_01.hpp>

#include "Bolt.hpp"

namespace LMM {

  class PhenoBuilder {

  public:

    enum SnpRegionType {
      CAUSAL_REGION, // chrom 1st-halves (pre-buffer): used in LD Score regression in simulations
      CANDIDATE_SNP, // simulated candidate snp with fixed large effect sizes
      BUFFER_REGION, // buffer in middle of chrom (not used for anything)
      NULL_REGION_H2HI_CHROMS, // 2nd-halves (post-buffer) of hi-h2 chroms: used to check avg null
      NULL_REGION_H2LO_CHROMS, // 2nd-halves (post-buffer) of lo-h2 chroms: used to check avg null
      BAD_REGION // masked SNPs (not used for anything)
    };
    enum EffectDistType {
      GAUSSIAN,
      LAPLACE
    };

  private:

    const Bolt &bolt;
    uint64 Nstride;
    EffectDistType effectDist;

    boost::mt19937 rng;
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > randn;
    boost::variate_generator<boost::mt19937&, boost::uniform_01<> > rand;
    boost::variate_generator<boost::mt19937&, boost::exponential_distribution<> > exprnd;

    // build masked, mean-centered, normalized genetic component of phenotype from input betas
    std::vector <double> buildPhenoGenetic(const std::vector <double> &betas);
    void maskCenterNormalize(std::vector <double> &pheno);

  public:

    static const int DEFAULT_MID_CHROM_HALF_BUFFER_PHYSPOS;
    static const double DEFAULT_MID_CHROM_HALF_BUFFER_GENPOS;

    // note: the Bolt instance should apply any additional masking needed (indivs, snps)
    PhenoBuilder(const Bolt &_bolt, uint seed, EffectDistType _effectDist);

    static bool isNullRegion(SnpRegionType snpRegion);

    /**
     * build masked, mean-centered, normalized genetic component of phenotype
     *
     * params:
     * - pCausal: prob of a 1st-half chrom being causal, BEFORE further reduction from MAF matching
     * - MAFhistFile: file containing allele freq spectrum to match
     * - stdPow: MAF-dependence; 0 for equal per-normalized-genotype, 1 for equal per-allele
    */
    std::vector <double> genPhenoCausal(std::vector <double> &simBetas,
					std::vector <SnpRegionType> &simRegions, int vcNum,
					int Mcausal, const std::string &MAFhistFile, double stdPow,
					std::set <int> highH2Chroms,
					int midChromHalfBufferPhyspos);

    std::vector <double> genPhenoCausalRegions(std::vector <double> &simBetas,
					       std::vector <SnpRegionType> &simRegions,
					       double lambdaRegion, double pRegion);
    /**
     * build masked, mean-centered, normalized candidate snp genetic component of phenotype
     * update simRegions accordingly to identify chosen candidate snps
     */
    std::vector <double> genPhenoCandidate(std::vector <SnpRegionType> &simRegions,
					   int Mcandidate);

    // build masked, mean-centered, normalized stratification component of phenotype
    std::vector <double> genPhenoStrat(const std::string &phenoStratFile);

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
    std::vector <double> genPhenoEnv(void);

    std::vector <double> combinePhenoComps
    (const std::vector <double> &h2causal, double h2candidate, double h2strat,
     const std::vector < std::vector <double> > &phenoCausal,
     const std::vector <double> &phenoCandidate, const std::vector <double> &phenoStrat,
     const std::vector <double> &phenoEnv);
  };
}

#endif
