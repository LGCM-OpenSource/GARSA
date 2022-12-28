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

#ifndef SNPDATA_HPP
#define SNPDATA_HPP

#include <vector>
#include <string>
#include <map>
#include <boost/utility.hpp>

#include "Types.hpp"
#include "SnpInfo.hpp"
#include "RestrictSnpSet.hpp"
#include "FileUtils.hpp"

namespace LMM {

  class SnpData : boost::noncopyable {

  public:

    static const uint64 IND_MISSING;

  private:
    uint64 Mbed, Nbed; // original PLINK data dimensions
    std::vector <int> bedIndivToRemoveIndex;
    std::vector <int> bedSnpToGrmIndex; // -2 = exclude, -1 = non-GRM, >=0 = GRM

    // M = # of snps in .bim file not in --exclude file and not failing QC
    // N = # of indivs in .fam file not in --remove file
    // Nstride = stride length of genotype lines stored in memory
    uint64 M, N, Nstride; // post-remove (but pre-QC) indivs, post-exclude and post-QC snps
    uchar *genotypes; // [[MATRIX]]: M x Nstride/4 (PLINK bed format, restricted to M x N)

    // maskSnps is all-1s M-vector (currently unused; keep in case masking is eventually needed)
    // maskIndivs is zero-filled to Nstride; only indivs failing missingness QC are masked
    uchar *maskSnps; // [VECTOR]: M (0 = ignore)
    double *maskIndivs; // [VECTOR]: Nstride (0 = ignore); set by QC (max % missing)
    uint64 numIndivsQC; // number of indivs remaining after QC = sum(maskIndivs)

    std::vector <SnpInfo> snps; // [VECTOR]: M
    bool mapAvailable;
    int Nautosomes;
    
    std::map <std::string, uint64> FID_IID_to_ind;
  
    struct IndivInfo {
      std::string famID;
      std::string indivID;
      std::string paternalID;
      std::string maternalID;
      int sex; // (1=male; 2=female; other=unknown)
      double pheno;
    };
    std::vector <IndivInfo> indivs; // [VECTOR]: N
    
    std::vector <std::string> vcNames; // names of variance comps (1-indexed; ignore 0th entry)

    /**
     * work: 256x4 aligned work array
     * lookupBedCode[4] = {value of 0, value of missing, value of 1, value of 2}
     */
    void buildByteLookup(double (*work)[4], const double lookupBedCode[4]) const;

    // subroutines for fast chip LD Score estimation
    void estChipLDscoresChrom(std::vector <double> &chipLDscores, int chrom,
			      const std::vector <int> &indivs) const;
    static float dotProdToAdjR2(float dotProd, int n);
    bool fillSnpSubRowNorm1(float x[], uint64 m, const std::vector <int> &indivs) const;
    

    void processIndivs(const std::string &famFile, const std::vector <std::string> &removeFiles);
    std::vector <SnpInfo> processSnps(std::vector <uint64> &Mfiles,
				      const std::vector <std::string> &bimFiles,
				      const std::vector <std::string> &excludeFiles,
				      const std::vector <std::string> &modelSnpsFiles,
				      const std::vector <std::string> &vcNamesIn,
				      bool loadNonModelSnps);
    void processMap(std::vector <SnpInfo> &bedSnps, const std::string &geneticMapFile,
		    bool noMapCheck);
    void storeBedLine(uchar bedLineOut[], const uchar genoLine[]);

  public:
    /**    
     * reads indiv info from fam file, snp info from bim file
     * allocates memory, reads genotypes, and does QC
     * assumes numbers of bim and bed files match
     */
    SnpData(const std::string &famFile, const std::vector <std::string> &bimFiles,
	    const std::vector <std::string> &bedFiles, const std::string &geneticMapFile,
	    const std::vector <std::string> &excludeFiles,
	    const std::vector <std::string> &modelSnpsFiles,
	    const std::vector <std::string> &removeFiles,
	    double maxMissingPerSnp, double maxMissingPerIndiv, bool noMapCheck,
	    std::vector <std::string> vcNamesIn=std::vector <std::string> (),
	    bool loadNonModelSnps=true, int _Nautosomes=22);

    ~SnpData();

    void freeGenotypes();

    static int chrStrToInt(std::string chrom, int Nauto);
    static std::vector <SnpInfo> readBimFile(const std::string &bimFile, int Nauto);
    static bool dosageValid(double dosage);

    /**
     * assumes Nbed and bedIndivToRemoveIndex have been initialized
     * if loadGenoLine == false, just advances the file pointer
     */
    void readBedLine(uchar genoLine[], uchar bedLineIn[], FileUtils::AutoGzIfstream &fin,
		     bool loadGenoLine) const;
    double computeAlleleFreq(const uchar genoLine[], const double subMaskIndivs[]) const;
    double computeAlleleFreq(const double genoLine[], const double subMaskIndivs[]) const;
    double computeMAF(const uchar genoLine[], const double subMaskIndivs[]) const;
    double computeSnpMissing(const uchar genoLine[], const double subMaskIndivs[]) const;
    double computeSnpMissing(const double dosageLine[], const double subMaskIndivs[]) const;
    // assumes maskedSnpVector has dimension Nstride; zero-fills
    void genoLineToMaskedSnpVector(double maskedSnpVector[], const uchar genoLine[],
				   const double subMaskIndivs[], double MAF) const;
    // assumes maskedSnpVector has dimension Nstride; zero-fills
    void dosageLineToMaskedSnpVector(double dosageLineVec[], const double subMaskIndivs[],
				     double MAF) const;

    uint64 getM(void) const;
    // don't provide getN: don't want the rest of the program to even know N!
    uint64 getNstride(void) const;
    uint64 getNbed(void) const;
    const double* getMaskIndivs(void) const;
    uint64 getNumIndivsQC(void) const;
    void writeMaskSnps(uchar out[]) const;
    void writeMaskIndivs(double out[]) const;
    const std::vector <SnpInfo> &getSnpInfo(void) const;
    const std::vector <int> &getBedSnpToGrmIndex(void) const;
    std::vector <double> getFamPhenos(void) const;    
    bool getMapAvailable(void) const;
    const uchar* getGenotypes(void) const;
    int getNumVCs(void) const;
    std::vector <std::string> getVCnames(void) const;
    
    uint64 getIndivInd(std::string &FID, std::string &IID) const;
  
    // check same chrom and within EITHER gen or phys distance
    bool isProximal(uint64 m1, uint64 m2, double genWindow, int physWindow) const;

    // note: pheno may have size Nstride, but we only output N values
    void writeFam(const std::string &outFile, const std::vector <double> &pheno) const;

    /*
    // creates map <string, double> and goes through snpInfo
    // plain text, 2-col: rs, LDscore; # to ignore
    void loadLDscore(const string &file) {
    // clear existing LDscore (set to NaN/-1?)
    }
    // use LDscore regression to calibrate stats[]
    void calibrateStats(const double stats[]) const {

    }
    */
  
    // writes Nstride values to out[]
    // assumes subMaskIndivs is a sub-mask of maskIndivs
    //   (presumably obtained by using writeMaskIndivs and taking a subset)
    // work: 256x4 aligned work array
    void buildMaskedSnpVector(double out[], const double subMaskIndivs[], uint64 m,
			      const double lut0129[4], double (*work)[4]) const;
    
    /**
     * performs fast rough computation of LD Scores among chip snps to use in regression weights
     * approximations:
     * - only uses a subsample of individuals
     * - replaces missing genotypes with means and uses them in dot product
     * - computes adjusted r^2 using 1/n correction instead of baseline
     */
    std::vector <double> estChipLDscores(uint64 sampleSize) const;


    // TODO: make private
    // note n <= N because of potential missing data/masking
    double compute_r2(const int x[], const int y[], int N) const;

    // fill Nstride-element array with 0, 1, 2, 9; missing or mask => 9
    void fillSnpRow(int x[], uint64 m) const;

    bool augmentLDscores(uint64 m, const std::vector <int> &mRow, uint64 mp, const std::vector <int> &mpRow,
			 const std::vector < std::pair <double, int> > &windows,
			 const std::vector <double> &alphaMAFdeps, std::vector <double> &LDscores,
			 std::vector <double> &windowCounts) const;

    std::vector <double> computeLDscore(uint64 m, const std::vector < std::pair <double, int> > &windows,
				   std::vector <double> &windowCounts, double &baselineR2,
				   const RestrictSnpSet &restrictPartnerSet,
				   const std::vector <double> &alphaMAFdeps) const;
  };
}

#endif
