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

#include <iostream>
#include <cstdlib>
#include <cmath>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "MatrixUtils.hpp"

namespace MatrixUtils {

  namespace ublas = boost::numeric::ublas;
  using std::cout;
  using std::cerr;
  using std::endl;

  ublas::matrix <double> chol(ublas::matrix <double> A) {
    int n = A.size1();
    ublas::matrix <double> L = ublas::zero_matrix<double>(n, n);
    for (int i = 0; i < n; i++)
      for (int j = 0; j <= i; j++) {
	for (int k = 0; k < j; k++)
	  A(i, j) -= L(i, k) * L(j, k);
	if (j == i)
	  L(i, i) = sqrt(A(i, i));
	else
	  L(i, j) = A(i, j) / L(j, j);
      }
    return L;
  }

  double minCholDiagSq(ublas::matrix <double> A) {
    int n = A.size1();
    double ret = 1e100;
    ublas::matrix <double> L = ublas::zero_matrix<double>(n, n);
    for (int i = 0; i < n; i++)
      for (int j = 0; j <= i; j++) {
	for (int k = 0; k < j; k++)
	  A(i, j) -= L(i, k) * L(j, k);
	if (j == i) {
	  ret = std::min(ret, A(i, i));
	  if (ret <= 0) return ret;
	  L(i, i) = sqrt(A(i, i));
	}
	else
	  L(i, j) = A(i, j) / L(j, j);
      }
    ret = std::min(ret, A(n-1, n-1));
    return ret;
  }

  ublas::matrix <double> invert(ublas::matrix <double> A) {
    ublas::permutation_matrix<std::size_t> lu_perm(A.size1());
    if (ublas::lu_factorize(A, lu_perm) != 0) {
      cerr << "ERROR: Matrix not invertible" << endl;
      exit(1);
    }
    ublas::matrix <double> Ainv = ublas::identity_matrix<double>(A.size1());
    ublas::lu_substitute(A, lu_perm, Ainv);
    return Ainv;
  }

  ublas::vector <double> linSolve(ublas::matrix <double> A, ublas::vector <double> b) {
    int n = b.size();
    ublas::permutation_matrix<std::size_t> lu_perm(n);
    if (ublas::lu_factorize(A, lu_perm) != 0) {
      cerr << "ERROR: Matrix not invertible (linSolve)" << endl;
      exit(1);
    }
    ublas::lu_substitute(A, lu_perm, b);
    return b;
  }

}
