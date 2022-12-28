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
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "zlib.h"

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

#include "SnpData.hpp"
#include "StringUtils.hpp"
#include "FileUtils.hpp"
#include "Types.hpp"

namespace FileUtils {

  using std::string;
  using std::vector;
  using std::cout;
  using std::cerr;
  using std::endl;

  void openOrExit(std::ifstream &stream, const string &file,
		  std::ios_base::openmode mode) {
    stream.open(file.c_str(), mode);
    if (!stream) {
      cerr << "ERROR: Unable to open file: " << file << endl;
      exit(1);
    }
  }
  void openWritingOrExit(std::ofstream &stream, const string &file,
			 std::ios_base::openmode mode) {
    stream.open(file.c_str(), mode);
    if (!stream) {
      cerr << "ERROR: Unable to open file for writing: " << file << endl;
      exit(1);
    }
  }
  void requireEmptyOrReadable(const std::string &file) {
    if (file.empty()) return;
    std::ifstream fin;
    fin.open(file.c_str());
    if (!fin) {
      cerr << "ERROR: Unable to open file: " << file << endl;
      exit(1);
    }
    fin.close();
  }  
  void requireEachEmptyOrReadable(const std::vector <std::string> &fileList) {
    for (uint i = 0; i < fileList.size(); i++)
      requireEmptyOrReadable(fileList[i]);
  }
  void requireEmptyOrWriteable(const std::string &file) {
    if (file.empty()) return;
    std::ofstream fout;
    fout.open(file.c_str(), std::ios::out|std::ios::app);
    if (!fout) {
      cerr << "ERROR: Output file is not writeable: " << file << endl;
      exit(1);
    }
    fout.close();
  }
  vector <string> parseHeader(const string &fileName, const string &delimiters) {
    AutoGzIfstream fin; fin.openOrExit(fileName);
    string header;
    getline(fin, header);
    vector <string> split = StringUtils::tokenizeMultipleDelimiters(header, delimiters);
    fin.close();
    return split;
  }
  int lookupColumnInd(const string &fileName, const string &delimiters, const string &columnName) {
    vector <string> headers = parseHeader(fileName, delimiters);
    int columnInd = -1;
    for (uint c = 0; c < headers.size(); c++)
      if (headers[c] == columnName)
	columnInd = c; // first column is snp ID, treated separately
    if (columnInd == -1) {
      cerr << "WARNING: Column " << columnName << " not found in headers of " << fileName << endl;
      //exit(1);
    }
    return columnInd;
  }
  double readDoubleNanInf(std::istream &stream) {
    string str;
    stream >> str;
    double x;
    sscanf(str.c_str(), "%lf", &x);
    return x;
  }

  vector < std::pair <string, string> > readFidIids(const string &file) {
    vector < std::pair <string, string> > ret;
    AutoGzIfstream fin;
    fin.openOrExit(file);
    string FID, IID, line;
    while (fin >> FID >> IID) {
      if (FID.empty() || IID.empty()) {
	cerr << "ERROR: In file " << file << endl;
	cerr << "       unable to read FID and IID; check format" << endl;
	exit(1);
      }
      ret.push_back(make_pair(FID, IID));
      getline(fin, line);
    }
    fin.close();
    return ret;
  }

  vector < std::pair <string, string> > readSampleIDs(const string &file) {
    vector < std::pair <string, string> > ret;
    AutoGzIfstream fin;
    fin.openOrExit(file);
    string FID, IID, line;
    int lineNum = 0;
    while (fin >> FID >> IID) {
      lineNum++;
      if (FID.empty() || IID.empty()) {
	cerr << "ERROR: In file " << file << endl;
	cerr << "       unable to read FID and IID; check format" << endl;
	exit(1);
      }
      if (lineNum == 1) {
	if (FID != "ID_1" || IID != "ID_2") {
	  cerr << "ERROR: In sample file " << file << endl;
	  cerr << "       first line must begin with ID_1 ID_2" << endl;
	  exit(1);
	}
      }
      else if (lineNum == 2) {
	if (FID != "0" || IID != "0") {
	  cerr << "ERROR: In sample file " << file << endl;
	  cerr << "       second line must begin with 0 0" << endl;
	  exit(1);
	}
      }
      else
	ret.push_back(make_pair(FID, IID));
      getline(fin, line);
    }
    fin.close();
    return ret;
  }

  int AutoGzIfstream::lineCount(const std::string &file) {
    AutoGzIfstream finFile; finFile.openOrExit(file);
    int ctr = 0; string line;
    while (getline(finFile, line))
      ctr++;
    finFile.close();
    return ctr;
  }

  /***** AutoGzIfstream class implementation *****/

  void AutoGzIfstream::openOrExit(const std::string &file, std::ios_base::openmode mode) {
    fin.open(file.c_str(), mode);
    if (!fin) {
      cerr << "ERROR: Unable to open file: " << file << endl;
      exit(1);
    }
    if ((int) file.length() > 3 && file.substr(file.length()-3) == ".gz")
      boost_in.push(boost::iostreams::gzip_decompressor());
    boost_in.push(fin);
  }

  void AutoGzIfstream::close() {
    fin.close();
    boost_in.reset();
  }

  AutoGzIfstream::operator bool() const {
    return !boost_in.fail();
  }

  AutoGzIfstream& AutoGzIfstream::read(char *s, std::streamsize n) {
    boost_in.read(s, n);
    return *this;
  }

  std::streamsize AutoGzIfstream::gcount() const {
    return boost_in.gcount();
  }

  int AutoGzIfstream::get() {
    return boost_in.get();
  }

  double AutoGzIfstream::readDoubleNanInf() {
    return FileUtils::readDoubleNanInf(boost_in);
  }

  void AutoGzIfstream::clear() {
    boost_in.clear();
  }

  AutoGzIfstream& AutoGzIfstream::seekg(std::streamoff off, std::ios_base::seekdir way) {
    boost_in.seekg(off, way);
    return *this;
  }

  AutoGzIfstream& getline(AutoGzIfstream& in, std::string &s) {
    std::getline(in.boost_in, s);
    return in;
  }

  
  /***** AutoGzOfstream class implementation *****/

  void AutoGzOfstream::openOrExit(const std::string &file, std::ios_base::openmode mode) {
    fout.open(file.c_str(), mode);
    if (!fout) {
      cerr << "ERROR: Unable to open file: " << file << endl;
      exit(1);
    }
    if ((int) file.length() > 3 && file.substr(file.length()-3) == ".gz")
      boost_out.push(boost::iostreams::gzip_compressor());
    boost_out.push(fout);
  }

  void AutoGzOfstream::close() {
    boost_out.reset();
    fout.close();
  }

  AutoGzOfstream& AutoGzOfstream::operator << (std::ostream&(*manip)(std::ostream&)) {
    manip(boost_out);
    return *this;
  }

  void AutoGzOfstream::unsetf(std::ios_base::fmtflags mask) {
    boost_out.unsetf(mask);
  }

  AutoGzOfstream::operator bool() const {
    return !boost_out.fail();
  }

  int checkBgenSample(const std::string &bgenFile, const std::string &sampleFile, int Nautosomes) {
    uint Nsample = FileUtils::readSampleIDs(sampleFile).size();

    cout << "Checking BGEN file " << bgenFile << endl;
    cout << "(with SAMPLE file " << sampleFile << ")..." << endl;

    FILE *fin = fopen(bgenFile.c_str(), "rb"); assert(fin != NULL);
    uint offset; fread_check(&offset, 4, 1, fin); //cout << "offset: " << offset << endl;
    uint L_H; fread_check(&L_H, 4, 1, fin); //cout << "L_H: " << L_H << endl;
    uint Mbgen; fread_check(&Mbgen, 4, 1, fin); cout << "snpBlocks (Mbgen): " << Mbgen << endl;
    assert(Mbgen != 0);
    uint Nbgen; fread_check(&Nbgen, 4, 1, fin); cout << "samples (Nbgen): " << Nbgen << endl;
    if (Nbgen != Nsample) {
      cerr << "ERROR: Number of samples in BGEN header does not match sample file" << endl;
      exit(1);
    }
    char magic[5]; fread_check(magic, 1, 4, fin); magic[4] = '\0'; //cout << "magic bytes: " << string(magic) << endl;
    fseek_check(fin, L_H-20, SEEK_CUR); //cout << "skipping L_H-20 = " << L_H-20 << " bytes (free data area)" << endl;
    uint flags; fread_check(&flags, 4, 1, fin); //cout << "flags: " << flags << endl;
    uint CompressedSNPBlocks = flags&3; cout << "CompressedSNPBlocks: " << CompressedSNPBlocks << endl;
    assert(CompressedSNPBlocks==1); // REQUIRE CompressedSNPBlocks==1
    uint Layout = (flags>>2)&0xf; cout << "Layout: " << Layout << endl;
    assert(Layout==1 || Layout==2); // REQUIRE Layout==1 or Layout==2

    //uint SampleIdentifiers = flags>>31; //cout << "SampleIdentifiers: " << SampleIdentifiers << endl;
    fseek_check(fin, offset+4, SEEK_SET);

    // check first SNP
    if (Layout==1) {
      uint Nrow; fread_check(&Nrow, 4, 1, fin); // cout << "Nrow: " << Nrow << " " << std::flush;
      if (Nrow != Nbgen) {
	cerr << "ERROR: Nrow = " << Nrow << " does not match Nbgen = " << Nbgen << endl;
	exit(1);
      }
    }
    char snpID[65536], rsID[65536], chrStr[65536], allele1[65536], allele0[65536];
    ushort LS; fread_check(&LS, 2, 1, fin); // cout << "LS: " << LS << endl;
    fread_check(snpID, 1, LS, fin); snpID[LS] = '\0'; cout << "first snpID: " << string(snpID) << endl;
    //fseek_check(fin, LS, SEEK_CUR); // skip SNP id
    ushort LR; fread_check(&LR, 2, 1, fin); // cout << "LR: " << LR << endl;
    fread_check(rsID, 1, LR, fin); rsID[LR] = '\0'; cout << "first rsID: " << string(rsID) << endl;

    if (Layout==2) {
      string snpName = string(rsID)=="." ? snpID : rsID;

      ushort LC; fread_check(&LC, 2, 1, fin); // cout << "LC: " << LC << " " << std::flush;
      fread_check(chrStr, 1, LC, fin); chrStr[LC] = '\0';

      int chrom = LMM::SnpData::chrStrToInt(chrStr, Nautosomes);
      if (chrom == -1) {
	cerr << "ERROR: Invalid chrom (expecting integer 1-" << Nautosomes+1
	     << " or X,XY,PAR1,PAR2): " << string(chrStr) << endl;
	exit(1);
      }

      uint bp; fread_check(&bp, 4, 1, fin); // cout << "bp: " << bp << " " << std::flush;

      ushort Kheader; fread_check(&Kheader, 2, 1, fin); //cout << "K: " << K << endl;
      if (Kheader != 2) {
	cerr << "ERROR: Non-bi-allelic variant found: " << Kheader << " alleles" << endl;
	exit(1);
      }

      uint LA; fread_check(&LA, 4, 1, fin); // cout << "LA: " << LA << " " << std::flush;
      fread_check(allele1, 1, LA, fin); allele1[LA] = '\0';

      uint LB; fread_check(&LB, 4, 1, fin); // cout << "LB: " << LB << " " << std::flush;
      fread_check(allele0, 1, LB, fin); allele0[LB] = '\0';

      uint C; fread_check(&C, 4, 1, fin); //cout << "C: " << C << endl;
      uchar *zBuf = (uchar *) malloc(C-4);
      uint D; fread_check(&D, 4, 1, fin); //cout << "D: " << D << endl;
      uchar *buf = (uchar *) malloc(D);
      fread_check(zBuf, 1, C-4, fin);

      /********** decompress and check genotype probability block **********/

      uLongf destLen = D, bufLen = D, zBufLen = C-4;
      //cout << "bufLen = " << bufLen << " zBufLen = " << zBufLen << endl;
      if (uncompress(buf, &destLen, zBuf, zBufLen) != Z_OK || destLen != bufLen) {
	cerr << "ERROR: uncompress() failed" << endl;
	exit(1);
      }
      uchar *bufAt = buf;
      uint N = bufAt[0]|(bufAt[1]<<8)|(bufAt[2]<<16)|(bufAt[3]<<24); bufAt += 4;
      if (N != Nbgen) {
	cerr << "ERROR: " << snpName << " has N = " << N << " (mismatch with header block)" << endl;
	exit(1);
      }
      uint K = bufAt[0]|(bufAt[1]<<8); bufAt += 2;
      if (K != 2U) {
	cerr << "ERROR: " << snpName << " has K = " << K << " (non-bi-allelic)" << endl;
	exit(1);
      }
      uint Pmin = *bufAt; bufAt++;
      if (Pmin != 2U) {
	cerr << "ERROR: " << snpName << " has minimum ploidy = " << Pmin << " (not 2)" << endl;
	exit(1);
      }
      uint Pmax = *bufAt; bufAt++;
      if (Pmax != 2U) {
	cerr << "ERROR: " << snpName << " has maximum ploidy = " << Pmax << " (not 2)" << endl;
	exit(1);
      }
      for (uint i = 0; i < N; i++) {
	uint ploidyMiss = *bufAt; bufAt++;
	if (ploidyMiss != 2U && ploidyMiss != 130U) {
	  cerr << "ERROR: " << snpName << " has ploidy/missingness byte = " << ploidyMiss
	       << " (not 2 or 130)" << endl;
	  exit(1);
	}
      }
      uint Phased = *bufAt; bufAt++;
      if (Phased != 0U && Phased != 1U) {
	cerr << "ERROR: " << snpName << " has Phased = " << Phased << " (not 0 or 1)" << endl;
	exit(1);
      }
      uint B = *bufAt; bufAt++;
      if (B != 8U) {
	cerr << "ERROR: " << snpName << " has B = " << B << " (not 8)" << endl;
	cerr << "       8-bit encoding is required for BGEN v1.2 files" << endl;
	exit(1);
      }

      free(buf);
      free(zBuf);
    }

    fclose(fin);

    cout << endl;

    return Layout;
  }

}
