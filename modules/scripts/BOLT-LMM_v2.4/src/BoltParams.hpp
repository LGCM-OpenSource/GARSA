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

#ifndef BOLTPARAMS_HPP
#define BOLTPARAMS_HPP

#include <vector>
#include <string>
#include <utility>

#include <boost/program_options.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include "DataMatrix.hpp"

namespace LMM {

  class BoltParams {
  public:

    static const double MIX_PARAM_ESTIMATE_FLAG; // flag for estimating mixture params using CV

    // main input files
    std::string famFile;
    std::vector <std::string> bimFiles, bedFiles; // possibly multiple files and/or range templates

    // optional reference map file for filling in genpos
    std::string geneticMapFile;

    std::vector <std::string> removeFiles; // list(s) of indivs to remove
    std::vector <std::string> excludeFiles; // list(s) of SNPs to exclude
    std::vector <std::string> modelSnpsFiles; // list(s) of SNPs to use in model (i.e., GRM)

    // QC params
    double maxMissingPerSnp, maxMissingPerIndiv;
    
    bool noMapCheck; // disable automatic check of genetic map scale
    int maxModelSnps; // error-check to discourage use of too many snps (e.g., imputed)

    // for real phenotype input
    std::string phenoFile;
    std::vector <std::string> phenoCols;
    bool phenoUseFam;

    // for real covariate input
    std::string covarFile;
    std::vector < std::pair <std::string, DataMatrix::ValueType> > covarCols;
    int covarMaxLevels;
    bool covarUseMissingIndic;

    // for analysis
    bool reml; // flag to run variance components analysis (automatic if computing assoc stats)
    bool lmmInf, lmmBayes, lmmBayesMCMC, lmmForceNonInf;
    double h2gGuess;
    int MCMCiters;
    int numLeaveOutChunks;
    int numCalibSnps;

    double pEst, varFrac2Est;
    int CVfoldsSplit, CVfoldsCompute;
    bool CVnoEarlyExit;

    int h2EstMCtrials, reEstMCtrials;
    int remlMCtrials;
    bool remlNoRefine;
    std::vector < boost::numeric::ublas::matrix <double> > remlGuessVegs;
    std::vector <std::string> remlGuessVCnames;
    bool runUnivarRemls;
    bool allowh2g01;

    // for avoiding proximal contamination
    double genWindow; int physWindow;

    // for calibration of lmmBayes[MCMC]
    std::string LDscoresFile, LDscoresCol, LDscoresChipCol;
    bool LDscoresUseChip;
    bool LDscoresMatchBp;
  
    // for stopping algorithm
    int maxIters;
    double CGtol, approxLLtol;
    
    int mBlockMultX, Nautosomes;
    
    int numThreads;

    // for final output
    std::string statsFile;
    bool verboseStats;
    std::string predBetasFile; // for (Bayesian) MLMi prediction

    // for dosage-format imputed SNPs
    std::vector <std::string> dosageFiles;
    std::string dosageFidIidFile;
    std::string statsFileDosageSnps;
    bool noDosageIDcheck;
    bool noDosage2IDcheck;
    bool noImpute2IDcheck;
    bool noBgenIDcheck;

    std::vector <std::string> dosage2MapFiles, dosage2GenoFiles;
    std::string statsFileDosage2Snps;

    std::vector <std::string> impute2Files;
    std::vector <int> impute2Chroms;
    std::string impute2FidIidFile;
    std::string statsFileImpute2Snps;
    double impute2MinMAF;

    std::vector <std::string> bgenFiles;
    std::vector <std::string> sampleFiles;
    std::string statsFileBgenSnps;
    std::vector <int> bgenLayouts; // 1 = v1.1, 2 = v1.2
    double bgenMinMAF, bgenMinINFO;
    bool domRecHetTest;

    // for output of simulated betas and chip LD Scores
    std::string snpInfoFile;

    // for PhenoBuilder
    uint seed;
    std::string MAFhistFile;
    int Mcausal, Mcandidate;
    double stdPow;
    int highH2ChromMax;
    int midChromHalfBufferPhyspos;
    std::string phenoStratFile;
    std::vector <double> h2causal;
    double h2candidate, h2strat;    
    double lambdaRegion, pRegion;
    std::string phenoOutFile;
    int effectDist;

    // populates members; error-checks
    bool processCommandLineArgs(int argc, char *argv[]);
  };
}

#endif
