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

#ifndef BOLT_HPP
#define BOLT_HPP

#include <vector>
#include <utility>
#include <boost/utility.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include "Types.hpp"
#include "SnpData.hpp"
#include "CovariateBasis.hpp"

namespace LMM {

  class Bolt : boost::noncopyable {

  public:
    static const double BAD_SNP_STAT;

    struct StatsDataRetroLOCO {
      std::string statName;
      std::vector <double> stats; // M-vector
      // projected and calibrated s.t. stat for snp x = dot(x / projNorm(x), resid[chunk])^2
      std::vector < std::vector <double> > calibratedResids; // # LOCO chunks x Nstride
      //std::vector <int> chunkAssignments; // M-vector; [0, # LOCO chunks) or -1 for masked snps
      std::vector < std::pair <uint64, int> > snpChunkEnds;
      std::vector <double> VinvScaleFactors; // Vinv*y / calibratedResids for each LOCO chunk
      StatsDataRetroLOCO(const std::string &_statName, const std::vector <double> &_stats,
			 const std::vector < std::vector <double> > &_calibratedResids,
			 const std::vector < std::pair <uint64, int> > &_snpChunkEnds,
			 const std::vector <double> &_VinvScaleFactors);
    };

  private:
    struct VarCompData {
      double logDelta, sigma2K, stdOfReml, stdToReml;
      std::vector <double> fJacks, fRandsAsData; // fJacks[MCtrials+1], fRandsAsData[MCtrials]
    };

#ifdef MEASURE_DGEMM
    mutable unsigned long long dgemmTicks;
#endif
    const SnpData &snpData; // use to obtain snp vectors via buildMaskedGenotypeVector
    const DataMatrix &covarDataT; // transposed covariate data matrix
    CovariateBasis covBasis; // contains covariates and maskIndivs
    const double *maskIndivs; // [VECTOR PTR]: NOT copied; set to point to covBasis.maskIndivs
    uint64 M, Nstride, Nused, Cindep, Cstride; // likewise inherited from snpData, covBasis

    double *Xnorm2s; // [VECTOR]: M (square norms of columns of X, i.e., normalized SNPs)
    double (*snpValueLookup)[4]; // [VECTOR]: M (4-tuples)
    double *snpCovBasisNegComps; // [[MATRIX]]: M x Cstride (zero-fill beyond Cindep)
    uchar *projMaskSnps; // [VECTOR]: M (a subset of snpData.maskSnps[]; local copy made)
    uint64 MprojMask; // number of snps left after masking = SUM(projMaskSnps)
    int numChromsProjMask; // number of chroms with >= 1 good snp
    double Xfro2; // squared Frobenius norm of X = SUM(Xnorm2s)

    uint64 mBlockMultX; // block size for X, X' mult in CG
    int Nautosomes;

    void init(void);
    uchar initMarker(uint64 m, double snpVector[]);

    inline void buildMaskedSnpNegCovCompVec(double snpCovCompVec[], uint64 m, double (*work)[4])
      const {
      snpData.buildMaskedSnpVector(snpCovCompVec, maskIndivs, m, snpValueLookup[m], work);
      // check: project out covariates immediately
      /*
      covBasis.projectCovars(snpCovCompVec);
      memset(snpCovCompVec + Nstride, 0, Cstride*sizeof(snpCovCompVec[0]));
      */
      // load in sign-flipped covar comps at snpCovBasisNegComps + m*Cstride
      memcpy(snpCovCompVec + Nstride, snpCovBasisNegComps + m*Cstride,
	     Cstride*sizeof(snpCovCompVec[0]));
    }

    inline void buildMaskedSnpCovCompVec(double snpCovCompVec[], uint64 m, double (*work)[4])
      const {
      buildMaskedSnpNegCovCompVec(snpCovCompVec, m, work);
      // sign-flip covar comps
      for (uint64 n = Nstride; n < Nstride+Cstride; n++)
	snpCovCompVec[n] *= -1;    
    }

    double dotCovCompVecs(const double xCovCompVec[], const double yCovCompVec[]) const;
    double computeProjNorm2(const double xCovCompVec[]) const;
    void computeProjNorm2s(double projNorm2s[], const double xCovCompVecs[], uint64 B) const;

    std::vector <int> makeChunkAssignments(int numLeaveOutChunks) const;
    std::vector < std::pair <uint64, int> >
    computeSnpChunkEnds(const std::vector <int> &chunkAssignments) const;
    int findChunkAssignment(const std::vector < std::pair <uint64, int> > &snpChunkEnds,
			    int chr, int bp) const;
    std::vector <uint64> makeBatchMaskSnps(uchar batchMaskSnps[],
					   const std::vector <int> &chunkAssignments,
					   const std::vector <int> &chunks, double genWindow,
					   int physWindow) const;
    std::vector <uint64> selectProSnps(int numCalibSnps, const double HinvPhiCovCompVec[],
				       int seed) const;

    void multXtrans(double XtransVecs[], const double vCovCompVecs[], uint64 B) const;
    void multX(double vCovCompVecs[], const double XtransVecs[], uint64 B) const;
    void multXXtransMask(double vCovCompVecs[], const uchar batchMaskSnps[], uint64 B) const;
    void multH(double HmultCovCompVecs[], const double xCovCompVecs[], const uchar batchMaskSnps[],
	       const double logDeltas[], const uint64 Ms[], uint64 B) const;
    void conjGradSolve(double xCovCompVecs[], bool useStartVecs, const double bCovCompVecs[],
		       const uchar batchMaskSnps[], const uint64 Ms[], const double logDeltas[],
		       uint64 B, int maxIters, double CGtol) const;
    void applySwaps(double x[], const std::vector < std::pair <uint64, uint64> > &swaps) const;
    void undoSwaps(double x[], const std::vector < std::pair <uint64, uint64> > &swaps) const;
    void swapCovCompVecs(double covCompVec1[], double covCompVec2[], double tmp[]) const;
    
    double logDeltaToH2(double logDelta) const;
    double h2ToLogDelta(double h2) const;
    void genMCscalingPhenoProjPairs(double yGenCovCompVecs[], double yEnvUnscaledCovCompVecs[],
				    std::vector <double> pheno, const uchar batchMaskSnps[],
				    const uint64 Ms[], uint64 B, int MCtrials, int seed) const;
    std::vector <double> computeMCscalingFs
    (/*double sigma2Ks[], */double HinvPhiCovCompVec[], VarCompData &testVCs/*double logDelta*/,
     const double yGenCovCompVecs[], const double yEnvUnscaledCovCompVecs[],
     const uchar batchMaskSnps[], const uint64 Ms[], uint64 B, int MCtrials, int CGmaxIters,
     double CGtol) const;
    void updateBestMCscalingF(VarCompData &bestVCs,/*double *sigma2Kbest, double *logDeltaBest, double *bestAbsF,*/
			      double HinvPhiCovCompVec[], VarCompData &testVCs/*double logDelta*/,
				const double yGenCovCompVecs[],
				const double yEnvUnscaledCovCompVecs[], int MCtrials,
				int CGmaxIters, double CGtol) const;
    void setMCtrials(int &MCtrials) const;

    void multThetaMinusIs(double multCovCompVecs[], const double xCovCompVecs[],
			  const uchar snpVCnums[], const std::vector <double> &vcXscale2s,
			  uint64 B, double coeffI=1.0) const;
    boost::numeric::ublas::vector <double> updateVegs
    (std::vector < boost::numeric::ublas::matrix <double> > &Vegs, double VinvyMultiCovCompVecs[],
     const uchar snpVCnums[], const std::vector <double> &vcXscale2s, int MCtrials) const;
    boost::numeric::ublas::vector <double> computeMCgrad
    (const double VinvyMultiCovCompVecs[], uint64 D, const uchar snpVCnums[], int VCs,
     const std::vector <double> &vcXscale2s, int MCtrials) const;
    boost::numeric::ublas::matrix <double> computeAI
    (const std::vector < boost::numeric::ublas::matrix <double> > &Vegs,
     const double VinvyMultiCovCompVecs[], const uchar snpVCnums[],
     const std::vector <double> &vcXscale2s, int CGmaxIters, double CGtol) const;
    void updateVegsAI(std::vector < boost::numeric::ublas::matrix <double> > &Vegs,
		      const uchar snpVCnums[], const std::vector <double> &vcXscale2s,
		      const double yRandsDataMultiCovCompVecs[], int MCtrials, int CGmaxIters,
		      double CGtol) const;
    double dotMultiCovCompVecs(const double xCovCompVecs[], const double yCovCompVecs[],
			       uint64 D) const;
    void computeMultiProjNorm2s(double projNorm2s[], const double xCovCompVecs[], uint64 D,
				uint64 B) const;
    void multVmulti(double VmultiCovCompVecs[], const double xMultiCovCompVecs[],
		    const uchar snpVCnums[],
		    const std::vector < boost::numeric::ublas::matrix <double> > &VegXscale2s,
		    uint64 B) const;
    void conjGradSolveVmulti(double xMultiCovCompVecs[], bool useStartVecs,
			     const double bMultiCovCompVecs[], uint64 B, const uchar snpVCnums[],
			     const std::vector <double> &vcXscale2s,
			     const std::vector < boost::numeric::ublas::matrix <double> > &Vegs,
			     int maxIters, double CGtol) const;
    void combineEnvGenMultiCovCompVecs
    (double yRandsDataMultiCovCompVecs[], const double yEnvGenUnscaledMultiCovCompVecs[],
     const std::vector < boost::numeric::ublas::matrix <double> > &Vegs, int MCtrials) const;
    boost::numeric::ublas::matrix <double> genUnscaledMultiCovCompVecs
    (double yEnvGenUnscaledMultiCovCompVecs[], const std::vector < std::vector <double> > &phenos,
     const uchar snpVCnums[], int VCs, const std::vector <double> &vcXscale2s, int MCtrials,
     int seed) const;

  public:
  
    /**
     * creates lookup tables of 0129 translation as well as covbasis components for each SNP
     * creates sub-mask to eliminate any bad snps
     */
    Bolt(const SnpData &_snpData, const DataMatrix &_covarDataT, const double _maskIndivs[],
	 const std::vector < std::pair <std::string, DataMatrix::ValueType> > &covars,
	 int covarMaxLevels, bool covarUseMissingIndic, int _mBlockMultX, int _Nautosomes);

    ~Bolt(void);

    const SnpData &getSnpData(void) const;
    const CovariateBasis &getCovBasis(void) const;
    const double *getMaskIndivs(void) const;
    const uchar *getProjMaskSnps(void) const;
    uint64 getMprojMask(void) const;
    int getNumChromsProjMask(void) const;
    uint64 getNused(void) const;
    uint64 getNCstride(void) const;

    void maskFillCovCompVecs(double covCompVecs[], const double vec[], uint64 B) const;
    // covComps: B x Cstride
    void fillCovComps(double covComps[], const double vec[], uint64 B) const;

    int batchComputeBayesIter
    (double yResidCovCompVecs[], double betasTrans[], const uchar batchMaskSnps[],
     const uint64 Ms[], const double logDeltas[], const double sigma2Ks[], double varFrac2Ests[],
     double pEsts[], uint64 B, bool MCMC, int maxIters, double approxLLtol) const;

    /**
     * pheno: needs to be copied (as it'll be extended to Nstride and projected)
     */
    StatsDataRetroLOCO computeLINREG(std::vector <double> pheno) const;

    /**
     * stat for snp m:
     * - compute ||phi_resid^LOCObatch[m]|| to normalize
     * - sq dot prod w/ SNP m, inc. covars: (x_m^T phi_resid^m / (||x_m|| * ||phi_resid^m||))^2
     */
    StatsDataRetroLOCO computeLmmBayes
    (std::vector <double> pheno, const std::vector <double> &logDeltas,
     const std::vector <double> &sigma2Ks, double varFrac2Est, double pEst, bool MCMC,
     double genWindow, int physWindow, int maxIters, double approxLLtol,
     const std::vector <double> &statsLmmInf, const std::vector <double> &LDscores,
     const std::vector <double> &LDscoresChip) const;

    /**
     * computes MLMe LOCO retrospective stat calibrated to prospective stat (like GRAMMAR-Gamma)
     * simultaneously solves linear systems H \ b for numerators and calibration snp denominators
     * uses previously-estimated h2 (logDeltaEst) and previously-computed Hinv*phi from VC est
     * - note that H in HinvPhi is genome-wide, not LOCO; ok for weeding outlier calibration snps
     *
     * stat: retrospective * (avg pro at random SNPs / avg retro at random SNPs)
     * retrospective stat: (N-C) * (x_m^T H_chunk[m]^-1 phi / (||x_m|| * ||H_chunk[m]^-1 phi||))^2
     * prospective stat: (N-C) * (x_m^T H_chunk[m]^-1 phi)^2
     *                   / (x_m^T H_chunk[m]^-1 x_m) / (phi^T H_chunk[m]^-1 phi)
     */
    StatsDataRetroLOCO computeLmmInf
    (std::vector <double> pheno, std::vector <double> logDeltas,
     const std::vector <double> &sigma2Ks, const double HinvPhiCovCompVec[], int numCalibSnps,
     double genWindow, int physWindow, int maxIters, double CGtol, int seed) const;

    /**
     * given betas, build phenotypes and collect negated covar comps
     * for out-of-sample prediction, apply covar coeffs to basis extension (to OOS indivs)
     *
     * phenoPreds: (out) B x Nstride
     * betasTrans: (in) M x B array of coefficients
     * fittedCovComps: (in) B x Cstride
     * extAllIndivs: 0 to apply maskIndivs, 1 to make predictions for all indivs
     */
    void batchComputePreds(double phenoPreds[], const double betasTrans[],
			   const double fittedCovComps[], uint64 B, bool extAllIndivs) const;

    /**
     * pheno: (in) B x Nstride
     * betasTrans: (in) M x B array of coefficients
     * predIndivs: mask; 1 for indivs to make predictions on
     */
    std::vector <double> batchComputePredPVEs(double *baselinePredMSEptr, const double pheno[],
					      const double betasTrans[], uint64 B,
					      const double predIndivs[]) const;

    /**
     * run VB/MCMC on all non-masked SNPs (no LOCO) to compute betas for MLMi prediction
     * write betas to file
     */
    void computeWritePredBetas(const std::string &betasFile, std::vector <double> pheno,
			       double logDelta, double sigma2K, double varFrac2Est, double pEst,
			       bool MCMC, int maxIters, double approxLLtol) const;

    /**
     * estimates log(delta) using all SNPs using secant method on MC scaling f_REML curve
     * also returns corresponding sigma2K and Hinv*phi (H using estimated log(delta)) for later use
     *
     * sigma2Kbest: (out) variance parameter for kinship (GRM) component
     * HinvPhiCovCompVec: (out) (Nstride+Cstride)-vector, allocated with aligned memory
     * pheno: (in) real phenotype, possibly of size N or zero-filled beyond (no covComps)
     */
    double estLogDelta(double *sigma2K, double HinvPhiCovCompVec[],
		       const std::vector <double> &pheno, int MCtrials,
		       double logDeltaTol, int CGmaxIters, double CGtol, int seed, bool allowh2g01)
      const;

    std::vector <double> reEstLogDeltas(const std::vector <double> &pheno, double logDeltaEst,
					const uchar batchMaskSnps[], const uint64 Ms[], uint64 B,
					int remlMCtrials, int CGmaxIters, double CGtol, int seed)
      const;
    /**
     * (in/out): logDeltas, sigma2Ks
     */
    void reEstVCs(std::vector <double> pheno, std::vector <double> &logDeltas,
		  std::vector <double> &sigma2Ks, int reEstMCtrials, double genWindow,
		  int physWindow, int maxIters, double CGtol, int seed) const;

    std::vector <double> estH2s(const std::vector <double> &pheno, const uchar snpVCnums[],
				const std::vector <double> &h2Guesses, int MCtrials,
				int CGmaxIters, double CGtol, int seed) const;
    void remlAI(std::vector < boost::numeric::ublas::matrix <double> > &Vegs, bool usePhenoCorrs,
		const std::vector < std::vector <double> > &phenos,
		const uchar snpVCnums[], int MCtrialsCoarse, int MCtrialsFine, int CGmaxIters,
		double CGtol, int seed) const;

    void printStatsHeader(FileUtils::AutoGzOfstream &fout, bool verboseStats, bool info,
			  const std::vector <StatsDataRetroLOCO> &retroData) const;
    std::string getSnpStats(const std::string &ID, int chrom, int physpos, double genpos,
			    const std::string &allele1, const std::string &allele0,
			    double alleleFreq, double missing, double workVec[], bool verboseStats,
			    const std::vector <StatsDataRetroLOCO> &retroData, double info=-9)
      const;
    std::string getSnpStats(const std::string &ID, int chrom, int physpos, double genpos,
			    const std::string &allele1, const std::string &allele0,
			    const uchar genoLine[], bool verboseStats,
			    const std::vector <StatsDataRetroLOCO> &retroData, double workVec[],
			    double info=-9)
      const;
    std::string getSnpStats(const std::string &ID, int chrom, int physpos, double genpos,
			    const std::string &allele1, const std::string &allele0,
			    double dosageLine[], bool verboseStats,
			    const std::vector <StatsDataRetroLOCO> &retroData, double info=-9)
      const;
    std::string getSnpStatsBgen2(uchar *buf, uint bufLen, const uchar *zBuf, uint zBufLen,
				 uint Nbgen, const std::vector <uint64> &bgenIndivInds,
				 const std::string &snpName, int chrom, int physpos, double genpos,
				 const std::string &allele1, const std::string &allele0,
				 double snpCovCompVec[], bool verboseStats,
				 const std::vector <StatsDataRetroLOCO> &retroData,
				 bool domRecHetTest, double bgenMinMAF, double bgenMinINFO)
      const;
    /**
     * compute retrospective LOCO assoc stats at all SNPs in input files
     * retroData contains calibrated residuals s.t.:
     *     stat for snp x = dot(x / projNorm(x), resid[chunk])^2
     * streams genotypes from input files and streams stats to output file
     */
    void streamComputeRetroLOCO
    (const std::string &outFile, const std::vector <std::string> &bimFiles,
     const std::vector <std::string> &bedFiles, const std::string &geneticMapFile,
     bool verboseStats, const std::vector <StatsDataRetroLOCO> &retroData) const;
    void streamDosages
    (const std::string &outFile, const std::vector <std::string> &dosageFiles,
     const std::string &dosageFidIidFile, const std::string &geneticMapFile, bool verboseStats,
     const std::vector <StatsDataRetroLOCO> &retroData) const;
    void streamDosage2
    (const std::string &outFile, const std::vector <std::string> &dosage2MapFiles,
     const std::vector <std::string> &dosage2GenoFiles, bool verboseStats,
     const std::vector <StatsDataRetroLOCO> &retroData) const;
    void streamImpute2
    (const std::string &outFile, const std::vector <std::string> &impute2Files,
     const std::vector <int> &impute2Chroms, const std::string &impute2FidIidFile,
     double impute2MinMAF, const std::string &geneticMapFile, bool verboseStats,
     const std::vector <StatsDataRetroLOCO> &retroData) const;
    void fastStreamImpute2
    (const std::string &outFile, const std::vector <std::string> &impute2Files,
     const std::vector <int> &impute2Chroms, const std::string &impute2FidIidFile,
     double impute2MinMAF, const std::string &geneticMapFile, bool verboseStats,
     const std::vector <StatsDataRetroLOCO> &retroData, bool domRecHetTest) const;
    void streamBgen
    (const std::string &outFile, int f, const std::string &bgenFile, const std::string &sampleFile,
     double bgenMinMAF, double bgenMinINFO, const std::string &geneticMapFile, bool verboseStats,
     const std::vector <StatsDataRetroLOCO> &retroData, bool domRecHetTest) const;
    void streamBgen2
    (const std::string &outFile, int f, const std::string &bgenFile, const std::string &sampleFile,
     double bgenMinMAF, double bgenMinINFO, const std::string &geneticMapFile, bool verboseStats,
     const std::vector <StatsDataRetroLOCO> &retroData, bool domRecHetTest, int threads) const;
  };
}

#endif
