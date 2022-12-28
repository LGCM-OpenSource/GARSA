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
#include <iostream>
#include <numeric>
#include <cmath>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "nlopt.hpp"

#include "MatrixUtils.hpp"
#include "NumericUtils.hpp"
#include "NonlinearOptMulti.hpp"

namespace NonlinearOptMulti {

  namespace ublas = boost::numeric::ublas;
  using std::cout;
  using std::cerr;
  using std::endl;

  struct FuncData {
    ublas::vector <double> ux, grad;
    ublas::matrix <double> AI;
    double maxStepNorm;
    FuncData(const ublas::vector <double> &_ux, const ublas::vector <double> &_grad,
	     const ublas::matrix <double> &_AI, double _maxStepNorm)
      : ux(_ux), grad(_grad), AI(_AI), maxStepNorm(_maxStepNorm) { }
  };

  double remlFunc(unsigned n, const double *x, double *grad, void *_data) {
    FuncData *data = (FuncData *) _data;
    const ublas::vector <double> &ux0 = data->ux;
    const ublas::vector <double> &grad0 = data->grad;
    const ublas::matrix <double> &AI = data->AI;
    ublas::vector <double> ux = ux0;
    for (unsigned i = 0; i < n; i++)
      ux(i) = x[i];
    if (grad) {
      ublas::vector <double> ugrad = grad0 - ublas::prod(AI, ux-ux0);
      for (unsigned i = 0; i < n; i++)
	grad[i] = ugrad(i);
    }
    return ublas::inner_prod(ux-ux0, grad0 - 0.5*(ublas::prod(AI, ux-ux0)));
  }

  double stepNormConstraint(unsigned n, const double *x, double *grad, void *_data) {
    const double mult = 1e6; // scale up so that constraint is respected under NLopt tolerances
    FuncData *data = (FuncData *) _data;
    const ublas::vector <double> &ux0 = data->ux;
    const ublas::matrix <double> &AI = data->AI;
    double maxStepNorm = data->maxStepNorm;
    if (grad)
      for (unsigned i = 0; i < n; i++)
	grad[i] = 2*(x[i]-ux0(i)) * NumericUtils::sq(AI(i, i)) * mult;
    double ret = -NumericUtils::sq(maxStepNorm);
    for (unsigned i = 0; i < n; i++)
      ret += NumericUtils::sq((x[i]-ux0(i)) * AI(i, i));
    return ret * mult;
  }

  double cholConstraint(unsigned n, const double *x, double *grad, void *_data) {
    const double mult = 1e6; // scale up so that constraint is respected under NLopt tolerances
    //cout << "cholCon"; for (unsigned i = 0; i < n; i++) cout << " " << x[i]; cout << endl; //x
    int *data = (int *) _data;
    uint64 D = data[0];
    int k = data[1];
    ublas::matrix <double> V = ublas::identity_matrix <double> (D);
    int curPar = k*D*(D+1)/2;
    for (uint64 di = 0; di < D; di++)
      for (uint64 dj = 0; dj <= di; dj++) {
	V(di, dj) = x[curPar++];
	V(dj, di) = V(di, dj);
      }
    //cout << "V" << V << endl; //x
    if (grad) {
      memset(grad, 0, n * sizeof(grad[0])); // pars for other VCs have no effect
      double dx = 1e-6;
      curPar = k*D*(D+1)/2;
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++) {
	  ublas::matrix <double> Vplus = V; Vplus(di, dj) = Vplus(dj, di) = V(di, dj) + dx;
	  ublas::matrix <double> Vminus = V; Vminus(di, dj) = Vminus(dj, di) = V(di, dj) - dx;
	  grad[curPar++] = (-MatrixUtils::minCholDiagSq(Vplus)
			    + MatrixUtils::minCholDiagSq(Vminus)) / (2*dx) * mult;
	}
      //cout << "grad"; for (unsigned i = 0; i < n; i++) cout << " " << grad[i]; cout << endl; //x
    }
    //cout << "ret " << 1e-9 - MatrixUtils::minCholDiagSq(V);
    return (1e-9 - MatrixUtils::minCholDiagSq(V)) * mult;
  }

  std::vector < ublas::matrix <double> > constrainedNR
  (double &dLLpred, ublas::vector <double> &p, const std::vector < ublas::matrix <double> > &Vegs,
   const ublas::vector <double> &grad, const ublas::matrix <double> &AI, double maxStepNorm) {

    int VCs = Vegs.size()-1;
    uint64 D = Vegs[0].size1();
    unsigned n = (1+VCs)*D*(D+1)/2;
    ublas::vector <double> ux(n);
    int curPar = 0;
    for (int k = 0; k <= VCs; k++)
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++)
	  ux(curPar++) = Vegs[k](di, dj);

    FuncData data(ux, grad, AI, maxStepNorm);
    
    std::vector <double> x0(n);
    for (unsigned i = 0; i < n; i++)
      x0[i] = ux(i);
    /*
    std::vector<double> lb(n), ub(n, 1);
    for (unsigned i = 0; i < n; i++) {
      if (F[i] < 0) {
	ub[i] = std::min(ub[i], x[i]);
      }
      else {
	lb[i] = std::max(lb[i], x[i]);
	bool signChange = false;
	for (double xt = x[i]; xt < 1; xt += 1e-5) {
	  double Ft = F[i] + A(i, i) * (xt*xt-x[i]*x[i]) + B(i, i) * (xt-x[i]);
	  if (Ft < 0) { // gradient went negative
	    ub[i] = std::min(ub[i], xt + 0.5 * (xt - x[i]));
	    signChange = true;
	    break;
	  }
	}
	if (!signChange) // limit at 3x
	  ub[i] = std::min(ub[i], std::max(1e-4, 3*x[i]));
	// also check Newton step
	double xt = -B(i, i) / (2*A(i, i));
	double Ft = F[i] + A(i, i) * (xt*xt-x[i]*x[i]) + B(i, i) * (xt-x[i]);
	if (A(i, i) > 0 && Ft > 0 && xt > x[i]) // grad still positive at parabola vertex: bound
	  ub[i] = std::min(ub[i], xt);
      }
      cout << "Bounds on coord " << i+1 << ": (" << lb[i]*lb[i] << ", " << ub[i]*ub[i] << ")"
	   << endl;
    }
    */
    std::vector<double> lb(n, -1e100);
    curPar = 0;
    for (int k = 0; k <= VCs; k++)
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++) {
	  if (di == dj)
	    lb[curPar] = 1e-9; // ensure positive diagonal entries
	  curPar++;
	}

    std::vector <double> xBest = x0; dLLpred = 0;
    for (int alg = 0; alg < 3; alg++) {
      nlopt::opt opt(alg==0 ? nlopt::LD_MMA : (alg==1 ? nlopt::LD_CCSAQ : nlopt::LD_SLSQP), n);
      opt.set_lower_bounds(lb);
      // opt.set_upper_bounds(ub);
      opt.set_max_objective(remlFunc, (void *) &data);
      opt.add_inequality_constraint(stepNormConstraint, (void *) &data, 1e-9);
      int cholConstraintPars[1+VCs][2];
      for (int k = 0; k < (int) Vegs.size(); k++) {
	cholConstraintPars[k][0] = (int) D;
	cholConstraintPars[k][1] = k;
	opt.add_inequality_constraint(cholConstraint, (void *) cholConstraintPars[k], 1e-10);
      }
      opt.set_xtol_rel(1e-6);
      opt.set_xtol_abs(std::vector <double> (n, 1e-6));

      try {
	std::vector <double> xTry = x0; double dLLpredTry;
	opt.set_maxeval(10000); // in rare cases, NLopt appears to go into an infinite loop
	opt.optimize(xTry, dLLpredTry);
	// make sure that variances are positive definite
	curPar = 0;
	for (int k = 0; k <= VCs; k++) {
	  ublas::matrix <double> Vdiag = ublas::identity_matrix <double> (D),
	    Voff = ublas::zero_matrix <double> (D, D);
	  for (uint64 di = 0; di < D; di++)
	    for (uint64 dj = 0; dj <= di; dj++) {
	      if (di == dj)
		Vdiag(di, di) = xTry[curPar];
	      else
		Voff(dj, di) = Voff(di, dj) = xTry[curPar];
	      curPar++;
	    }
	  double maxCorr = k == 0 ? 0.99 : 1; // limit environmental correlation to 0.9
	  double lo = 0, hi = 1; // binary search on factor to multiply off-diagonals by
	  for (int t = 0; t < 30; t++) {
	    double mid = (lo+hi)/2;
	    if (MatrixUtils::minCholDiagSq(maxCorr * Vdiag + mid * Voff) < 1e-9)
	      hi = mid;
	    else
	      lo = mid;
	  }
	  if (hi < 1)
	    cout << "Reducing off-diagonals by a factor of " << (1-hi)
		 << " to make matrix positive definite" << endl;
	  curPar -= D*(D+1)/2;
	  for (uint64 di = 0; di < D; di++)
	    for (uint64 dj = 0; dj <= di; dj++) {
	      if (di != dj)
		xTry[curPar] *= hi;
	      curPar++;
	    }
	}
	dLLpredTry = remlFunc(n, &xTry[0], NULL, (void *) &data);
	if (dLLpredTry > dLLpred) {
	  dLLpred = dLLpredTry;
	  xBest = xTry;
	}
      }
      catch (nlopt::roundoff_limited) {
	cerr << "WARNING: NLopt threw 'nlopt::roundoff_limited':" << endl;
	cerr << "         Optimization halted because roundoff errors limited progress" << endl;
      }
    }
    
    p = ublas::zero_vector <double> (n); // store step in output parameter
    for (unsigned i = 0; i < n; i++)
      p(i) = xBest[i] - ux(i);

    std::vector < ublas::matrix <double> > optVegs = Vegs;
    curPar = 0;
    for (int k = 0; k <= VCs; k++)
      for (uint64 di = 0; di < D; di++)
	for (uint64 dj = 0; dj <= di; dj++) {
	  optVegs[k](di, dj) = xBest[curPar++];
	  optVegs[k](dj, di) = optVegs[k](di, dj);
	}
    return optVegs;
  }
}
