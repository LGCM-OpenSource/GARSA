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

#ifndef NONLINEAROPTMULTI_HPP
#define NONLINEAROPTMULTI_HPP

#include <vector>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>

namespace NonlinearOptMulti {

  std::vector < boost::numeric::ublas::matrix <double> > constrainedNR
  (double &dLLpred, boost::numeric::ublas::vector <double> &p,
   const std::vector < boost::numeric::ublas::matrix <double> > &Vegs,
   const boost::numeric::ublas::vector <double> &grad,
   const boost::numeric::ublas::matrix <double> &AI,
   double maxStepNorm);

}

#endif
