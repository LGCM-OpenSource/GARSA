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
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <algorithm>
#include <utility>
#include <numeric>

#include "StringUtils.hpp"
#include "FileUtils.hpp"
#include "PhenoBuilder.hpp"
#include "MatrixUtils.hpp"
#include "BoltParams.hpp"

#include <boost/program_options.hpp>
#include <boost/numeric/ublas/matrix.hpp>


namespace LMM {

  using std::vector;
  using std::string;
  using std::cout;
  using std::cerr;
  using std::endl;
  namespace ublas = boost::numeric::ublas;

  vector < std::pair <string, string> > readFidIidsRemove(const string &famFile,
							  const vector <string> &removeFiles) {
    std::set < std::pair <string, string> > removeSet;
    for (uint f = 0; f < removeFiles.size(); f++) {
      vector < std::pair <string, string> > removeIDs = FileUtils::readFidIids(removeFiles[f]);
      removeSet.insert(removeIDs.begin(), removeIDs.end());
    }
    vector < std::pair <string, string> > plinkIDs = FileUtils::readFidIids(famFile);
    vector < std::pair <string, string> > plinkIDsRemove;
    for (uint i = 0; i < plinkIDs.size(); i++)
      if (removeSet.find(plinkIDs[i]) == removeSet.end())
	plinkIDsRemove.push_back(plinkIDs[i]);
    return plinkIDsRemove;
  }
  void writeMissingIndivs(vector < std::pair <string, string> > &plinkIDs,
			  vector < std::pair <string, string> > &imputedIDs) {
    std::set < std::pair <string, string> > imputedSet(imputedIDs.begin(), imputedIDs.end());
    int numMissing = 0;
    for (uint i = 0; i < plinkIDs.size(); i++)
      if (imputedSet.find(plinkIDs[i]) == imputedSet.end())
	numMissing++;
    char missingIndivFile[1000];
    sprintf(missingIndivFile, "bolt.in_plink_but_not_imputed.FID_IID.%d.txt", numMissing);

    cerr << "       Writing " << numMissing << " missing indivs to " << missingIndivFile << endl;
    cerr << "       (--remove this file to resolve the error)" << endl;
    std::ofstream fout(missingIndivFile);
    for (uint i = 0; i < plinkIDs.size(); i++)
      if (imputedSet.find(plinkIDs[i]) == imputedSet.end())
	fout << plinkIDs[i].first << " " << plinkIDs[i].second << endl;
    fout.close();
  }
  bool checkOverlap(const string &famFile,
		    const vector < std::pair <string, string> > &sortedIDs) {
    vector < std::pair <string, string> > plinkIDs = FileUtils::readFidIids(famFile);
    sort(plinkIDs.begin(), plinkIDs.end());
    vector < std::pair <string, string> > isectIDs;
    std::set_intersection(plinkIDs.begin(), plinkIDs.end(),
			  sortedIDs.begin(), sortedIDs.end(), std::back_inserter(isectIDs));
    return isectIDs.size() >= 0.5 * sortedIDs.size();
  }

  const double BoltParams::MIX_PARAM_ESTIMATE_FLAG = -1;

  // populates members; error-checks
  bool BoltParams::processCommandLineArgs(int argc, char *argv[]) {

    vector <string> bimFileTemplates, bedFileTemplates;
    vector <string> removeFileTemplates, excludeFileTemplates, modelSnpsFileTemplates;
    vector <string> covarColTemplates, qCovarColTemplates;
    vector <string> dosageFileTemplates, bgenFileTemplates;
    string dosage2FileList, impute2FileList;
    string remlGuessStr;
    string sampleFile1, bgenSampleFileList;

    namespace po = boost::program_options;
    po::options_description typical("Typical options");
    typical.add_options()
      ("help,h", "print help message with typical options")
      ("helpFull", "print help message with full option list")

      // genotype data parameters
      ("bfile", po::value<string>(), "prefix of PLINK .fam, .bim, .bed files")
      ("bfilegz", po::value<string>(), "prefix of PLINK .fam.gz, .bim.gz, .bed.gz files")
      ("fam", po::value<string>(&famFile),
       "PLINK .fam file (note: file names ending in .gz are auto-[de]compressed)")
      ("bim", po::value< vector <string> >(&bimFileTemplates)/*->multitoken()*/,
       "PLINK .bim file(s); for >1, use multiple --bim and/or {i:j}, e.g., data.chr{1:22}.bim")
      ("bed", po::value< vector <string> >(&bedFileTemplates)/*->multitoken()*/,
       "PLINK .bed file(s); for >1, use multiple --bim and/or {i:j} expansion")
      ("geneticMapFile", po::value<string>(&geneticMapFile),
       "Oxford-format file for interpolating genetic distances: tables/genetic_map_hg##.txt.gz")
      // "chr pos rate(cM/Mb) map(cM)"

      ("remove", po::value< vector <string> >(&removeFileTemplates),
       "file(s) listing individuals to ignore (no header; FID IID must be first two columns)")
      ("exclude", po::value< vector <string> >(&excludeFileTemplates),
       "file(s) listing SNPs to ignore (no header; SNP ID must be first column)")
      ("maxMissingPerSnp", po::value<double>(&maxMissingPerSnp)->default_value(0.1, "0.1"),
       "QC filter: max missing rate per SNP")
      ("maxMissingPerIndiv", po::value<double>(&maxMissingPerIndiv)->default_value(0.1, "0.1"),
       "QC filter: max missing rate per person")
      
      // phenotype and covariate data parameters
      ("phenoFile", po::value<string>(&phenoFile),
       "phenotype file (header required; FID IID must be first two columns)")
      ("phenoCol", po::value< vector <string> >(&phenoCols), "phenotype column header")
      ("phenoUseFam", "use last (6th) column of .fam file as phenotype")
      ("covarFile", po::value<string>(&covarFile),
       "covariate file (header required; FID IID must be first two columns)")
      ("covarCol", po::value< vector <string> >(&covarColTemplates),
       "categorical covariate column(s); for >1, use multiple --covarCol and/or {i:j} expansion")
      ("qCovarCol", po::value< vector <string> >(&qCovarColTemplates),
       "quantitative covariate column(s); for >1, use multiple --qCovarCol and/or {i:j} expansion")
      ("covarUseMissingIndic",
       "include samples with missing covariates in analysis via missing indicator method (default: ignore such samples)")

      // analysis parameters
      ("reml", "run variance components analysis to precisely estimate heritability (but not compute assoc stats)")
      ("lmm",
       "compute assoc stats under the inf model and with Bayesian non-inf prior (VB approx), if power gain expected")
      ("lmmInfOnly", "compute mixed model assoc stats under the infinitesimal model")
      ("lmmForceNonInf", "compute non-inf assoc stats even if BOLT-LMM expects no power gain")
      ("modelSnps", po::value< vector <string> >(&modelSnpsFileTemplates),
       "file(s) listing SNPs to use in model (i.e., GRM) (default: use all non-excluded SNPs)")

      // calibration parameters
      ("LDscoresFile", po::value<string>(&LDscoresFile),
       "LD Scores for calibration of Bayesian assoc stats: tables/LDSCORE.1000G_EUR.tab.gz")

      ("numThreads", po::value<int>(&numThreads)->default_value(1),
       "number of computational threads")
      ("statsFile", po::value<string>(&statsFile),
       "output file for assoc stats at PLINK genotypes")
      ("dosageFile", po::value< vector <string> >(&dosageFileTemplates),
       "file(s) containing imputed SNP dosages to test for association (see manual for format)")
      ("dosageFidIidFile", po::value<string>(&dosageFidIidFile),
       "file listing FIDs and IIDs of samples in dosageFile(s), one line per sample")
      ("statsFileDosageSnps", po::value<string>(&statsFileDosageSnps),
       "output file for assoc stats at dosage format genotypes")
      ("impute2FileList", po::value<string>(&impute2FileList),
       "list of [chr file] pairs containing IMPUTE2 SNP probabilities to test for association")
      ("impute2FidIidFile", po::value<string>(&impute2FidIidFile),
       "file listing FIDs and IIDs of samples in IMPUTE2 files, one line per sample")
      ("impute2MinMAF", po::value<double>(&impute2MinMAF)->default_value(0),
       "MAF threshold on IMPUTE2 genotypes; lower-MAF SNPs will be ignored")
      ("bgenFile", po::value< vector <string> >(&bgenFileTemplates),
       "file(s) containing Oxford BGEN-format genotypes to test for association")
      ("sampleFile", po::value<string>(&sampleFile1),
       "file containing Oxford sample file corresponding to BGEN file(s)")
      ("bgenSampleFileList", po::value<string>(&bgenSampleFileList),
       "list of [bgen sample] file pairs containing BGEN imputed variants to test for association")
      ("bgenMinMAF", po::value<double>(&bgenMinMAF)->default_value(0),
       "MAF threshold on Oxford BGEN-format genotypes; lower-MAF SNPs will be ignored")
      ("bgenMinINFO", po::value<double>(&bgenMinINFO)->default_value(0),
       "INFO threshold on Oxford BGEN-format genotypes; lower-INFO SNPs will be ignored")
      ("statsFileBgenSnps", po::value<string>(&statsFileBgenSnps),
       "output file for assoc stats at BGEN-format genotypes")
      ("statsFileImpute2Snps", po::value<string>(&statsFileImpute2Snps),
       "output file for assoc stats at IMPUTE2 format genotypes")
      ("dosage2FileList", po::value<string>(&dosage2FileList),
       "list of [map dosage] file pairs with 2-dosage SNP probabilities (Ricopili/plink2 --dosage format=2) to test for association")
      ("statsFileDosage2Snps", po::value<string>(&statsFileDosage2Snps),
       "output file for assoc stats at 2-dosage format genotypes")
      ;

    po::options_description additional("Additional options");
    additional.add_options()
      // error-checking
      ("noMapCheck", "disable automatic check of genetic map scale")
      ("noDosageIDcheck", "disable automatic check of match between PLINK and dosage sample IDs")
      ("noDosage2IDcheck", "disable automatic check of match between PLINK and 2-dosage sample IDs")
      ("noImpute2IDcheck", "disable automatic check of match between PLINK and IMPUTE2 sample IDs")
      ("noBgenIDcheck", "disable automatic check of match between PLINK and BGEN sample IDs")
      ("maxModelSnps", po::value<int>(&maxModelSnps)->default_value(1000000),
       "an error-check: if millions of SNPs are imputed, it's inefficient to use them all")
      ("covarMaxLevels", po::value<int>(&covarMaxLevels)->default_value(10),
       "an error-check: maximum number of levels for a categorical covariate")
      
      // detailed analysis parameters
      ("numLeaveOutChunks", po::value<int>(&numLeaveOutChunks)->default_value(-1),
       "# of SNP groups left out in turn to avoid proximal contamination (default: # chroms; LOCO analysis)")
      ("numCalibSnps", po::value<int>(&numCalibSnps)->default_value(30),
       "# of random SNPs at which to compute denominator of prospective statistic for calibration")
      ("h2gGuess", po::value<double>(&h2gGuess)->default_value(0.25),
       "initial guess of h2g for LMM assoc")
      ("h2EstMCtrials", po::value<int>(&h2EstMCtrials)->default_value(0),
       "number of MC trials to use when roughly estimating h2g for LMM assoc (0 = auto)")
      ("reEstMCtrials", po::value<int>(&reEstMCtrials)->default_value(0),
       "number of MC trials to use when re-estimating h2g for each LOCO rep (0 = no re-est)")
      ("remlNoRefine", "compute faster (~2-3x) but slightly less accurate (~1.03x higher SE) REML variance parameter estimates")
      ("remlGuessStr", po::value<string>(&remlGuessStr),
       "initial variance parameter guesses (see manual for format) for REML optimization")
      ("genWindow", po::value<double>(&genWindow)->default_value(0.02),
       "genetic dist buffer (Morgans) to avoid proximal contamination if # MLMe leave-out groups > # chroms")
      ("physWindow", po::value<int>(&physWindow)->default_value(2000000),
       "physical dist buffer (bp) to avoid proximal contamination if # MLMe leave-out groups > # chroms")
      ("pEst", po::value<double>(&pEst)->default_value(BoltParams::MIX_PARAM_ESTIMATE_FLAG),
       "prior prob SNP effect is drawn from large-effect mixture component (default: est via CV)")
      ("varFrac2Est", po::value<double>(&varFrac2Est)->
       default_value(BoltParams::MIX_PARAM_ESTIMATE_FLAG),
       "prior fraction of variance in small-effect mixture component (default: estimate via CV)")
      ("CVfoldsSplit", po::value<int>(&CVfoldsSplit)->default_value(5),
       "cross-validation folds to split samples into for mixture param estimation")
      ("CVfoldsCompute", po::value<int>(&CVfoldsCompute)->default_value(0),
       "max cross-validation folds to actually compute: for large N, few are needed (0 = auto)")
      ("CVnoEarlyExit",
       "run full CV (by default, CV exits once best param choice is statistically clear")
      ("LDscoresCol", po::value<string>(&LDscoresCol)->default_value("LDSCORE"),
       "column name of LD Scores to use in regression")
      ("LDscoresUseChip", "use LD Scores estimated among chip SNPs instead of reference panel")
      ("LDscoresMatchBp", "match SNPs to reference LD Scores based on (chr,bp) coordinates")
      ("Nautosomes", po::value<int>(&Nautosomes)->default_value(22),
       "number of autosomes for organism being studied")

      // numerical parameters
      ("CGtol", po::value<double>(&CGtol)->default_value(5e-4, "5e-4"),
       "tolerance for declaring convergence of conjugate gradient solver")
      ("approxLLtol", po::value<double>(&approxLLtol)->default_value(0.01),
       "tolerance for declaring convergence of variational Bayes")
      ("maxIters", po::value<int>(&maxIters)->default_value(500), "max number of iterations")
      ("snpsPerBlock", po::value<int>(&mBlockMultX)->default_value(64),
       "working set of SNPs to process at once while performing computations")

      // more analysis options
      ("lmmBayesMCMC", "compute Bayesian mixed model assoc stats using MCMC")
      ("MCMCiters", po::value<int>(&MCMCiters)->default_value(0),
       "number of MCMC iterations to use (default: min(maxIters, 5*number of VB iters from CV))")
      ("verboseStats", "output additional columns in statsFile")
      ("predBetasFile", po::value<string>(&predBetasFile),
       "output file of betas for risk prediction")
      ;

    // phenotype simulation parameters
    po::options_description hidden("Hidden options");
    hidden.add_options()
      ("allowX", "enable chrX analysis (now always enabled)")
      ("domRecHetTest", "run dominance/recessive/heterozygous advantage tests for BGEN SNPs")
      ("remlMCtrials", po::value<int>(&remlMCtrials)->default_value(15),
       "number of MC trials to use when estimating VCs for heritability analysis")
      ("runUnivarRemls", "run stochastic REML on multiple phenotypes individually in series")
      ("allowh2g01", "allow h2g estimates close to 0 or 1")
      ("seed", po::value<uint>(&seed)->default_value(0), "random seed")
      ("MAFhist", po::value<string>(&MAFhistFile), "file containing MAF histogram to match")
      ("Mcausal", po::value<int>(&Mcausal)->default_value(10000),
       "number of simulated causal SNPs")
      ("Mcandidate", po::value<int>(&Mcandidate)->default_value(0),
       "number of simulated candidate SNPs")
      ("stdPow", po::value<double>(&stdPow)->default_value(0),
       "MAF-dependent effect sizes: 0 eq per-SNP, 1 eq per-allele")
      ("highH2ChromMax", po::value<int>(&highH2ChromMax)->default_value(0),
       "chroms 1..highH2ChromMax share the [bulk of/all] simulated heritability")
      ("midChromHalfBufferPhyspos", po::value<int>(&midChromHalfBufferPhyspos)->
       default_value(PhenoBuilder::DEFAULT_MID_CHROM_HALF_BUFFER_PHYSPOS),
       "half-width of mid-chrom buffer separating causal region from null SNPs (-1 to turn off)")
      ("phenoStratFile", po::value<string>(&phenoStratFile), "vector (e.g., PC) to induce strat")
      ("h2causal", po::value< vector <double> >(&h2causal)/*->multitoken()*/,
       "heritability from causal SNPs (multiple values if multiple variance components)")
      ("h2candidate", po::value<double>(&h2candidate)->default_value(0),
       "heritability from candidate SNPs")
      ("h2strat", po::value<double>(&h2strat)->default_value(0), "heritability from pop labels")
      ("lambdaRegion", po::value<double>(&lambdaRegion)->default_value(0), "region scale (Mb)")
      ("pRegion", po::value<double>(&pRegion)->default_value(0), "fraction of causal regions")
      ("phenoOutFile", po::value<string>(&phenoOutFile), "output randomly generated phenotype")
      ("effectDist", po::value<int>(&effectDist)->default_value(0), "0 = Gaussian, 1 = Laplace")
      // TODO: hide?
      ("snpInfoFile", po::value<string>(&snpInfoFile),
       "extended output including simulated SNP effect sizes (in simulations) and chip LD Scores")
      ;

    po::options_description visible("Options");
    visible.add(typical).add(additional);

    po::options_description all("All options");
    all.add(typical).add(additional).add(hidden);
    all.add_options()
      ("bad-args", po::value< vector <string> >(), "bad args")
      ;
    po::positional_options_description positional_desc;
    positional_desc.add("bad-args", -1); // for error-checking command line
    
    po::variables_map vm;
    po::command_line_parser cmd_line(argc, argv);
    cmd_line.options(all);
    cmd_line.style(po::command_line_style::default_style ^ po::command_line_style::allow_guessing);
    cmd_line.positional(positional_desc);
    try {
      po::store(cmd_line.run(), vm);

      if (vm.count("help") || vm.count("helpFull")) {
	cout /*<< "BOLT-LMM computes..."*/ << endl;
	if (vm.count("helpFull"))
	  cout << visible << endl;
	else
	  cout << typical << endl;
	exit(0);
      }
      
      po::notify(vm); // throws an error if there are any problems

      if (vm.count("bfile") +
	  vm.count("bfilegz") +
	  (vm.count("fam") || vm.count("bim") || vm.count("bed")) != 1) {
	cerr << "ERROR: Use exactly one of the --bfile, --bfilegz, or --fam,bim,bed input formats"
	     << endl;
	if (!dosageFileTemplates.empty() || !impute2FileList.empty() || !bgenFileTemplates.empty())
	  cerr  << "       (even when analyzing imputed data, a plink file is needed for model-"
		<< "        fitting using a subset of SNPs, typically those directly genotyped)"
		<< endl;
	return false;
      }

      if (vm.count("bfile")) {
	string bfile = vm["bfile"].as<string>();
	famFile = bfile + ".fam";
	bimFileTemplates.push_back(bfile + ".bim");
	bedFileTemplates.push_back(bfile + ".bed");
      }

      if (vm.count("bfilegz")) {
	string bfile = vm["bfilegz"].as<string>();
	famFile = bfile + ".fam.gz";
	bimFileTemplates.push_back(bfile + ".bim.gz");
	bedFileTemplates.push_back(bfile + ".bed.gz");
      }

      phenoUseFam = vm.count("phenoUseFam");
      noMapCheck = vm.count("noMapCheck");
      if (vm.count("allowX"))
	cerr << "NOTE: --allowX is now always set; there is no need to set this flag" << endl;
      CVnoEarlyExit = vm.count("CVnoEarlyExit");

      covarUseMissingIndic = vm.count("covarUseMissingIndic");

      reml = vm.count("reml");
      lmmForceNonInf = vm.count("lmmForceNonInf");
      lmmBayes = vm.count("lmm") || lmmForceNonInf;
      lmmBayesMCMC = vm.count("lmmBayesMCMC");
      lmmInf = vm.count("lmmInfOnly") || lmmBayes || lmmBayesMCMC; // Bayes needs inf for calib
      LDscoresUseChip = vm.count("LDscoresUseChip");
      LDscoresMatchBp = vm.count("LDscoresMatchBp");
      verboseStats = vm.count("verboseStats");
      remlNoRefine = vm.count("remlNoRefine");
      runUnivarRemls = vm.count("runUnivarRemls");
      allowh2g01 = vm.count("allowh2g01");
      noDosageIDcheck = vm.count("noDosageIDcheck");
      noDosage2IDcheck = vm.count("noDosage2IDcheck");
      noImpute2IDcheck = vm.count("noImpute2IDcheck");
      noBgenIDcheck = vm.count("noBgenIDcheck");

      domRecHetTest = vm.count("domRecHetTest");
      
      if (vm.count("bad-args")) {
	cerr << "ERROR: Unknown options:";
	vector <string> bad_args = vm["bad-args"].as< vector <string> >();
	for (uint i = 0; i < bad_args.size(); i++) cerr << " " << bad_args[i];
	cerr << endl;
	return false;
      }

      if (famFile.empty()) {
	cerr << "ERROR: fam file must be specified either using --fam or --bfile"
	     << endl;
	return false;
      }
      if (bimFileTemplates.empty()) {
	cerr << "ERROR: bim file(s) must be specified either using --bim or --bfile"
	     << endl;
	return false;
      }
      if (bedFileTemplates.empty()) {
	cerr << "ERROR: bed file(s) must be specified either using --bed or --bfile"
	     << endl;
	return false;
      }
      if (bimFileTemplates.size() != bedFileTemplates.size()) {
	cerr << "ERROR: Numbers of bim files and bed files must match" << endl;
	return false;
      }
      bimFiles = StringUtils::expandRangeTemplates(bimFileTemplates);
      bedFiles = StringUtils::expandRangeTemplates(bedFileTemplates);
      if (bimFiles.size() != bedFiles.size()) {
	cerr << "ERROR: Numbers of bim files and bed files must match" << endl;
	return false;
      }
      removeFiles = StringUtils::expandRangeTemplates(removeFileTemplates);
      excludeFiles = StringUtils::expandRangeTemplates(excludeFileTemplates);
      modelSnpsFiles = StringUtils::expandRangeTemplates(modelSnpsFileTemplates);
      if (phenoFile.empty() != phenoCols.empty()) {
	cerr << "ERROR: --phenoFile and --phenoCol" << endl
	     << "       must either be both specified or both unspecified"
	     << endl;
	return false;
      }
      if (!phenoFile.empty() && phenoUseFam) {
	cerr << "ERROR: If --phenoFile is specified, --phenoUseFam cannot be set" << endl;
	return false;
      }
      if (phenoFile.empty() && !phenoUseFam && seed==0) {
	cerr << "ERROR: Either {--phenoFile,--phenoCol} or --phenoUseFam must be set" << endl;
	return false;
      }
      if (!covarFile.empty()) {
	if (covarColTemplates.empty() && qCovarColTemplates.empty()) {
	  cerr << "ERROR: If --covarFile is specified, >=1 --covarCol or --qCovarCol is required"
	       << endl;
	  return false;
	}
	else {
	  vector <string> covarColVec = StringUtils::expandRangeTemplates(covarColTemplates);
	  vector <string> qCovarColVec = StringUtils::expandRangeTemplates(qCovarColTemplates);
	  for (uint64 i = 0; i < covarColVec.size(); i++)
	    covarCols.push_back(std::make_pair(covarColVec[i], DataMatrix::CATEGORICAL));
	  for (uint64 i = 0; i < qCovarColVec.size(); i++)
	    covarCols.push_back(std::make_pair(qCovarColVec[i], DataMatrix::QUANTITATIVE));
	}
      }
      else {
	if (!covarColTemplates.empty() || !qCovarColTemplates.empty()) {
	  cerr << "ERROR: If --covarCol or --qCovarCol(s) are specified, --covarFile is required"
	       << endl;
	  return false;
	}
      }

      if (!(0 <= maxMissingPerSnp && maxMissingPerSnp <= 1)) {
	cerr << "ERROR: --maxMissingPerSnp must be between 0 and 1" << endl;
	return false;	
      }
      if (!(0 <= maxMissingPerIndiv && maxMissingPerIndiv <= 1)) {
	cerr << "ERROR: --maxMissingPerIndiv must be between 0 and 1" << endl;
	return false;	
      }

      if (remlNoRefine && !reml) {
	cerr << "ERROR: --remlNoRefine can only be specified if --reml is specified" << endl;
	return false;
      }
      if (reml && lmmInf) {
	cerr << "ERROR: --reml and --lmm* options cannot both be specified" << endl;
	return false;
      }
      if (lmmInf && phenoCols.size() > 1) {
	cerr << "ERROR: Only one --phenoCol may be specified for association analysis" << endl;
	return false;
      }
      if (lmmInf && statsFile.empty()) {
	cerr << "ERROR: --statsFile must be specified for association output" << endl;
	return false;
      }
      if (vm.count("predBetasFile") && !lmmInf) {
	cerr << "ERROR: --predBetasFile requires specifying one of the --lmm* options" << endl;
	return false;
      }
      if (lmmBayes || lmmBayesMCMC) {
	if (LDscoresUseChip) {
	  //if (!LDscoresCol.empty())
	  //  cerr << "WARNING: Ignoring --LDscoresCol because --LDscoresUseChip is set" << endl;
	  if (!LDscoresFile.empty()) {
	    cerr << "ERROR: If --LDscoresUseChip is set, --LDscoresFile cannot be set"
		 << endl;
	    return false;
	  }
	}
	else {
	  if (LDscoresFile.empty() || LDscoresCol.empty()) {
	    cerr << "ERROR: LDscoresFile and LDscoresCol required to calibrate Bayes assoc stats"
		 << endl << "(unless --LDscoresUseChip is set)" << endl;
	    return false;
	  }
	}
      }
      else {
	if (pEst != BoltParams::MIX_PARAM_ESTIMATE_FLAG)
	  cerr << "WARNING: Ignoring pEst parameter (n/a for infinitesimal model)"
	       << endl;	
	if (varFrac2Est != BoltParams::MIX_PARAM_ESTIMATE_FLAG)
	  cerr << "WARNING: Ignoring varFrac2Est parameter (n/a for infinitesimal model)"
	       << endl;	
      }
      if (h2gGuess <= 0 || h2gGuess >= 1) {
	cerr << "ERROR: --h2gGuess must be between 0 and 1" << endl;
	return false;
      }

      // expand and error-check dosage files (simple dosage format)
      int numDosageParams =
	!dosageFileTemplates.empty() + !dosageFidIidFile.empty() + !statsFileDosageSnps.empty();
      if (numDosageParams != 0 && numDosageParams != 3) {
	cerr << "ERROR: --dosageFile(s), --dosageFidIidFile, and --statsFileDosageSnps must either be all specified or all not specified" << endl;
	return false;
      }
      if (numDosageParams) {
	vector < std::pair <string, string> > plinkIDs = readFidIidsRemove(famFile, removeFiles);
	vector < std::pair <string, string> > dosageIDs = FileUtils::readFidIids(dosageFidIidFile);
	sort(plinkIDs.begin(), plinkIDs.end());
	sort(dosageIDs.begin(), dosageIDs.end());
	vector < std::pair <string, string> > isectIDs;
	std::set_intersection(plinkIDs.begin(), plinkIDs.end(),
			      dosageIDs.begin(), dosageIDs.end(), std::back_inserter(isectIDs));
	if (isectIDs.size() < plinkIDs.size()) {
	  cerr << "ERROR: Some samples in --famFile/bfile are missing in --dosageFidIidFile"
	       << endl;
	  writeMissingIndivs(plinkIDs, dosageIDs);
	  return false;
	}
	if (!checkOverlap(famFile, dosageIDs)) {
	  if (noDosageIDcheck)
	    cerr << "WARNING: Overlap between --dosageFidIidFile and --famFile is < 50%" << endl;
	  else {
	    cerr << "ERROR: Overlap between --dosageFidIidFile and --famFile is < 50%" << endl;
	    cerr << "       (to override and perform the analysis anyway, set --noDosageIDcheck)"
		 << endl;
	    return false;
	  }
	}
	int Ndosage = dosageIDs.size();
	dosageFiles = StringUtils::expandRangeTemplates(dosageFileTemplates);
	for (uint64 i = 0; i < dosageFiles.size(); i++) {
	  FileUtils::AutoGzIfstream fin;
	  fin.openOrExit(dosageFiles[i]);
	  string line; getline(fin, line);
	  std::istringstream iss(line);
	  string rsID; iss >> rsID;
	  string chromStr;
	  if (!(iss >> chromStr) || SnpData::chrStrToInt(chromStr, Nautosomes) == -1) {
	    cerr << "ERROR: In --dosageFile " << dosageFiles[i] << endl;
	    cerr << "       unable to read chrom number as second token; check format" << endl;
	    return false;
	  }
	  int pos; if (!(iss >> pos)) {
	    cerr << "ERROR: In --dosageFile " << dosageFiles[i] << endl;
	    cerr << "       unable to read base pair pos as third token; check format" << endl;
	    return false;
	  }
	  string token; int ctr = 0;
	  while (iss >> token) ctr++;
	  if (ctr != Ndosage+2) {
	    cerr << "ERROR: In --dosageFile " << dosageFiles[i] << endl;
	    cerr << "       wrong number of entries in first line:" << endl;
	    cerr << "       expected: rsID, chr, pos, A1, A0, and N=" << Ndosage << " dosages"
		 << endl;
	    cerr << "       read: " << ctr+3 << " entries" << endl;
	    return false;
	  }
	  fin.close();
	}
      }

      // expand and error-check dosage files (IMPUTE2 format)
      int numImpute2Params =
	!impute2FileList.empty() + !impute2FidIidFile.empty() + !statsFileImpute2Snps.empty();
      if (numImpute2Params != 0 && numImpute2Params != 3) {
	cerr << "ERROR: --impute2FileList, --impute2FidIidFile, and --statsFileImpute2Snps" << endl
	     << "        must either be all specified or all unspecified"
	     << endl;
	cerr << "  impute2FileList: " << impute2FileList << endl;
	cerr << "  impute2FidIidFile: " << impute2FidIidFile << endl;
	cerr << "  statsFileImpute2Snps: " << statsFileImpute2Snps << endl;
	return false;
      }
      if (numImpute2Params) {
	vector < std::pair <string, string> > plinkIDs = readFidIidsRemove(famFile, removeFiles);
	vector < std::pair <string, string> > impute2IDs =
	  FileUtils::readFidIids(impute2FidIidFile);
	sort(plinkIDs.begin(), plinkIDs.end());
	sort(impute2IDs.begin(), impute2IDs.end());
	vector < std::pair <string, string> > isectIDs;
	std::set_intersection(plinkIDs.begin(), plinkIDs.end(),
			      impute2IDs.begin(), impute2IDs.end(), std::back_inserter(isectIDs));
	if (isectIDs.size() < plinkIDs.size()) {
	  cerr << "ERROR: Some samples in --famFile/bfile are missing in --impute2FidIidFile"
	       << endl;
	  writeMissingIndivs(plinkIDs, impute2IDs);
	  return false;
	}
	if (!checkOverlap(famFile, impute2IDs)) {
	  if (noImpute2IDcheck)
	    cerr << "WARNING: Overlap between --impute2FidIidFile and --famFile is < 50%" << endl;
	  else {
	    cerr << "ERROR: Overlap between --impute2FidIidFile and --famFile is < 50%" << endl;
	    cerr << "       (to override and perform the analysis anyway, set --noImpute2IDcheck)"
		 << endl;
	    return false;
	  }
	}
	int Nimpute2 = impute2IDs.size();

	FileUtils::AutoGzIfstream finList; finList.openOrExit(impute2FileList);
	string chromStr, file; int lineNum = 1;
	while (finList >> chromStr >> file) {
	  string junk; getline(finList, junk);
	  int chrom = SnpData::chrStrToInt(chromStr, Nautosomes);
	  if (chrom == -1) {
	    cerr << "ERROR: Invalid chrom in field 1, line " << lineNum << " of --impute2FileList:"
		 << endl
		 << "       " << chromStr << " " << file << " " << junk << endl;
	    return false;
	  }
	  FileUtils::requireEmptyOrReadable(file);
	  impute2Files.push_back(file);
	  impute2Chroms.push_back(chrom);
	  lineNum++;
	}
	finList.close();

	for (uint64 i = 0; i < impute2Files.size(); i++) {
	  FileUtils::AutoGzIfstream fin;
	  fin.openOrExit(impute2Files[i]);
	  string line; getline(fin, line);
	  std::istringstream iss(line);
	  string snpID, rsID; iss >> snpID >> rsID;
	  int pos; if (!(iss >> pos)) {
	    cerr << "ERROR: In --impute2File " << impute2Files[i] << endl;
	    cerr << "       unable to read base pair pos as third token; check format" << endl;
	    return false;
	  }
	  string token; int ctr = 0;
	  while (iss >> token) ctr++;
	  if (ctr != 3*Nimpute2+2) {
	    cerr << "ERROR: In --impute2File " << impute2Files[i] << endl;
	    cerr << "       wrong number of entries in first line:" << endl;
	    cerr << "       expected: snpID, rsID, pos, A1, A0, and 3*N = " << 3*Nimpute2
		 << " impute2 probabilities" << endl;
	    cerr << "       read: " << ctr+3 << " entries" << endl;
	    return false;
	  }
	  fin.close();
	}
      }

      // error-check dosage files (BGEN format)
      if (statsFileBgenSnps.empty() &&
	  !(bgenFileTemplates.empty() && sampleFile1.empty() && bgenSampleFileList.empty())) {
	cerr << "ERROR: --statsFileBgenSnps is required if --bgenFile(s)/--sampleFile or --bgenSampleFileList is specified" << endl;
	return false;
      }
      if (!statsFileBgenSnps.empty()) {
	// error-check for invalid param combinations;
	// create vectors of corresponding bgen/sample files
	if (!bgenSampleFileList.empty()) { // --bgenSampleFileList provided
	  if (!(bgenFileTemplates.empty() && sampleFile1.empty())) {
	    cerr << "ERROR: --bgenFile(s)/--sampleFile and --bgenSampleFileList cannot both be specified" << endl;
	    return false;
	  }
	  cout << "Verifying contents of --bgenSampleFileList: " << bgenSampleFileList << endl;	  
	  FileUtils::AutoGzIfstream finList; finList.openOrExit(bgenSampleFileList);
	  string bFile, sFile;
	  while (finList >> bFile >> sFile) {
	    FileUtils::requireEmptyOrReadable(bFile); bgenFiles.push_back(bFile);
	    FileUtils::requireEmptyOrReadable(sFile); sampleFiles.push_back(sFile);
	  }
	  finList.close();
	}
	else { // --bgenSampleFileList not provided
	  if (bgenFileTemplates.empty() || sampleFile1.empty()) {
	    cerr << "ERROR: --bgenFile(s) and --sampleFile must both be specified if --statsFileBgenSnps is specified and --bgenSampleFileList is not"
		 << endl;
	    return false;
	  }
	  bgenFiles = StringUtils::expandRangeTemplates(bgenFileTemplates);
	  for (uint64 f = 0; f < bgenFiles.size(); f++)
	    FileUtils::requireEmptyOrReadable(bgenFiles[f]);
	  FileUtils::requireEmptyOrReadable(sampleFile1);
	  sampleFiles = vector <string> (bgenFiles.size(), sampleFile1);
	}

	// error-check each bgen/sample file pair; save layouts
	vector < std::pair <string, string> > plinkIDs = readFidIidsRemove(famFile, removeFiles);
	sort(plinkIDs.begin(), plinkIDs.end());
	bgenLayouts.resize(bgenFiles.size());
	for (uint64 f = 0; f < bgenFiles.size(); f++) {
	  string sampleFile = sampleFiles[f];
	  vector < std::pair <string, string> > sampleIDs = FileUtils::readSampleIDs(sampleFile);
	  sort(sampleIDs.begin(), sampleIDs.end());
	  vector < std::pair <string, string> > isectIDs;
	  std::set_intersection(plinkIDs.begin(), plinkIDs.end(),
				sampleIDs.begin(), sampleIDs.end(), std::back_inserter(isectIDs));
	  if (isectIDs.size() < plinkIDs.size()) {
	    cerr << "ERROR: Some samples in --famFile/bfile are missing in sample file:" << endl
		 << "       " << sampleFile << endl;
	    writeMissingIndivs(plinkIDs, sampleIDs);
	    return false;
	  }
	  if (!checkOverlap(famFile, sampleIDs)) {
	    if (noBgenIDcheck)
	      cerr << "WARNING: Overlap is < 50% between --famFile and sample file:" << endl
		   << "         " << sampleFile << endl;
	    else {
	      cerr << "ERROR: Overlap is < 50% between --famFile and sample file:" << endl
		   << "       " << sampleFile << endl;
	      cerr << "       (to override and perform the analysis anyway, set --noBgenIDcheck)"
		   << endl;
	      return false;
	    }
	  }
	  bgenLayouts[f] = FileUtils::checkBgenSample(bgenFiles[f], sampleFile, Nautosomes);
	}
      }

      // expand and error-check dosage files (Ricopili 2-dosage format)
      int numDosage2Params = !dosage2FileList.empty() + !statsFileDosage2Snps.empty();
      if (numDosage2Params != 0 && numDosage2Params != 2) {
	cerr << "ERROR: --dosage2FileList and --statsFileDosage2Snps" << endl
	     << "       must either be both specified or both unspecified"
	     << endl;
	return false;
      }
      if (numDosage2Params) {
	cout << "Verifying contents of --dosage2FileList: " << dosage2FileList << endl;
	FileUtils::AutoGzIfstream finList, finMap, finGeno; finList.openOrExit(dosage2FileList);
	string mapFile, genoFile;
	while (finList >> mapFile >> genoFile) {
	  dosage2MapFiles.push_back(mapFile);
	  dosage2GenoFiles.push_back(genoFile);
	  cout << "Checking map file " << mapFile << " and 2-dosage genotype file " << genoFile
	       << endl;
	  finMap.openOrExit(mapFile);
	  string mapLine; getline(finMap, mapLine);
	  finMap.close();
	  finGeno.openOrExit(genoFile);
	  string genoHeader; getline(finGeno, genoHeader);
	  string genoLine; getline(finGeno, genoLine);
	  finGeno.close();
	  std::istringstream issMapLine(mapLine), issGenoHeader(genoHeader), issGenoLine(genoLine);

	  // check basic format of 1st line of map file
	  string chromStr, snpIDmap; double genpos; int physpos;
	  issMapLine >> chromStr;
	  int chrom = SnpData::chrStrToInt(chromStr, Nautosomes);
	  if (chrom == -1) {
	    cerr << "ERROR: Invalid chrom in field 1, line 1 of map file " << mapFile << ":"
		 << "       " << chromStr << endl;
	    return false;
	  }
	  issMapLine >> snpIDmap;
	  if (!(issMapLine >> genpos)) {
	    cerr << "ERROR: Unable to read genetic pos as field 3 of map file " << mapFile << endl;
	    return false;
	  }
	  if (!(issMapLine >> physpos)) {
	    cerr << "ERROR: Unable to read bp position as field 4 of map file " << mapFile << endl;
	    return false;
	  }
	  
	  // check header of 2-dosage geno file
	  string SNP, A1, A2;
	  issGenoHeader >> SNP >> A1 >> A2;
	  if (SNP != "SNP" || A1 != "A1" || A2 != "A2") {
	    cerr << "ERROR: 2-dosage genotype file does not begin with 'SNP A1 A2': " << genoFile
		 << endl;
	    return false;
	  }
	  vector < std::pair <string, string> > dosage2IDs; string FID, IID;
	  while (issGenoHeader >> FID >> IID)
	    dosage2IDs.push_back(std::make_pair(FID, IID));
	  vector < std::pair <string, string> > plinkIDs = readFidIidsRemove(famFile, removeFiles);
	  sort(plinkIDs.begin(), plinkIDs.end());
	  sort(dosage2IDs.begin(), dosage2IDs.end());
	  vector < std::pair <string, string> > isectIDs;
	  std::set_intersection(plinkIDs.begin(), plinkIDs.end(), dosage2IDs.begin(),
				dosage2IDs.end(), std::back_inserter(isectIDs));
	  if (isectIDs.size() < plinkIDs.size()) {
	    cerr << "ERROR: Some samples in --famFile/bfile are missing in dosage2 geno file "
		 << genoFile << endl;
	    writeMissingIndivs(plinkIDs, dosage2IDs);
	    return false;
	  }
	  if (!checkOverlap(famFile, dosage2IDs)) {
	    if (noDosage2IDcheck)
	      cerr << "WARNING: Overlap between dosage2 geno file and --famFile is < 50%" << endl;
	    else {
	      cerr << "ERROR: Overlap between dosage2 geno file and --famFile is < 50%" << endl;
	      cerr << "       (to override and perform analysis anyway, set --noDosage2IDcheck)"
		   << endl;
	      return false;
	    }
	  }
	  int Ndosage2 = dosage2IDs.size();
	  
	  // check basic format of 1st line of 2-dosage geno file
	  issGenoLine >> SNP >> A1 >> A2;
	  if (SNP != snpIDmap) {
	    cerr << "ERROR: SNP ID of 1st line of geno file (after header) does not match map file"
		 << endl;
	    return false;
	  }
	  double prob; int ctr = 0;
	  while (issGenoLine >> prob) ctr++;
	  if (ctr != 2*Ndosage2) {
	    cerr << "ERROR: In dosage2 geno file " << genoFile << endl;
	    cerr << "       wrong number of entries in first line:" << endl;
	    cerr << "       expected: SNP, A1, A2, and 2*N = " << 2*Ndosage2
		 << " dosage2 probabilities" << endl;
	    cerr << "       read: " << ctr << " probabilities" << endl;
	    return false;
	  }
	}
	finList.close();
	cout << endl;
      }

      if (!remlGuessStr.empty()) {
	cout << "Parsing --remlGuessStr: \"" << remlGuessStr << "\"" << endl << endl;
	int D = runUnivarRemls || phenoCols.empty() ? 1 : phenoCols.size();
	cout << "Number of phenotype vectors to (jointly) analyze: D = " << D << endl;
	cout << "Number of parameters per variance component: D*(D+1)/2 = "
	     << D*(D+1)/2 << endl;
	printf("Expecting: env/noise [%d args] [vcName1] [%d args] [vcName2] [%d args] ...\n",
	       D*(D+1)/2, D*(D+1)/2, D*(D+1)/2);
	cout << "  (if all SNPs are in 1 variance component, use 'modelSnps' for vcName1)" << endl;
	cout << "Each upper triangle should contain h2s on diagonal, corrs above diagonal"
	     << endl;
	cout << "  e.g., for 3 traits: h^2_1 r_12 r_13 h^2_2 r_23 h^2_3" << endl;
	std::istringstream iss(remlGuessStr);
	string vcName;
	while (iss >> vcName) {
	  if (remlGuessVCnames.empty() && vcName != "env/noise") {
	    cerr << "ERROR: First token of --remlGuessStr must be 'env/noise'" << endl;
	    return false;
	  }
	  remlGuessVCnames.push_back(vcName);
	  ublas::matrix <double> V = ublas::identity_matrix <double> (D);
	  for (int i = 0; i < D; i++)
	    for (int j = i; j < D; j++) {
	      double x;
	      if (!(iss >> x)) {
		cerr << "ERROR: Failed to read " << D*(D+1)/2
		     << " numerical value(s) for VC named '" << vcName << "'" << endl;
		return false;
	      }
	      V(i, j) = V(j, i) = x;
	    }
	  // check psd
	  if (MatrixUtils::minCholDiagSq(V) < 1e-9) {
	    cerr << "ERROR: Cov matrix for '" << vcName << "' is not strictly pos definite"
		 << endl;
	    return false;
	  }
	  remlGuessVegs.push_back(V);
	}
	// check heritabilities sum to 1
	for (int i = 0; i < D; i++) {
	  double h2sum = 0;
	  for (int k = 0; k < (int) remlGuessVegs.size(); k++)
	    h2sum += remlGuessVegs[k](i, i);
	  if (fabs(h2sum-1) > 1e-3) {
	    cerr << "ERROR: Sum of covariance matrices must have all diagonal entries = 1"
		 << endl;
	    cerr << "       Phenotype " << (i+1) << " has sum(h2) = " << h2sum << endl;
	    return false;
	  }
	}
	cout << endl;
      }

      if (genWindow > 0.1) {
	cerr << "ERROR: Max genetic window allowed is 0.1 Morgans (10 cM)" << endl;
	return false;
      }
      if (h2causal.empty())
	h2causal.push_back(0.5);
      if (h2strat != 0 && phenoStratFile.empty()) {
	cerr << "ERROR: If h2strat != 0, phenoStratFile must be specified" << endl;
	return false;
      }
      if (std::accumulate(h2causal.begin(), h2causal.end(), 0.0) + h2candidate + h2strat > 1) {
	cerr << "ERROR: sum(h2causal) + h2candidate + h2strat > 1" << endl;
	return false;
      }

      if (highH2ChromMax==0) highH2ChromMax = Nautosomes;

      // check that all files specified are readable/writeable
      FileUtils::requireEmptyOrReadable(famFile);
      FileUtils::requireEachEmptyOrReadable(bimFiles);
      FileUtils::requireEachEmptyOrReadable(bedFiles);
      FileUtils::requireEmptyOrReadable(geneticMapFile);
      FileUtils::requireEachEmptyOrReadable(removeFiles);
      FileUtils::requireEachEmptyOrReadable(excludeFiles);
      FileUtils::requireEachEmptyOrReadable(modelSnpsFiles);
      FileUtils::requireEmptyOrReadable(phenoFile);
      FileUtils::requireEmptyOrReadable(covarFile);
      FileUtils::requireEmptyOrReadable(LDscoresFile);
      FileUtils::requireEmptyOrWriteable(statsFile);
      FileUtils::requireEmptyOrWriteable(predBetasFile);
      FileUtils::requireEmptyOrWriteable(snpInfoFile);
      FileUtils::requireEmptyOrWriteable(statsFileDosageSnps);
      FileUtils::requireEmptyOrWriteable(statsFileImpute2Snps);

      FileUtils::requireEmptyOrReadable(MAFhistFile);
      FileUtils::requireEmptyOrReadable(phenoStratFile);
      FileUtils::requireEmptyOrWriteable(phenoOutFile);
    }
    catch (po::error &e) {
      cerr << "ERROR: " << e.what() << endl << endl;
      cerr << visible << endl;
      return false;
    }
    return true;
  }
}

