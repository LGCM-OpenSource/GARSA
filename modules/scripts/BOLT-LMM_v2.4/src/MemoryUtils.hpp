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

#ifndef MEMORYUTILS_HPP
#define MEMORYUTILS_HPP

#include "Types.hpp"

#define MEM_ALIGNMENT 64

//#define ALIGNED_MALLOC(size) mkl_malloc(size, MEM_ALIGNMENT)
//#define ALIGNED_MALLOC(size) _mm_malloc(size, MEM_ALIGNMENT)
void *ALIGNED_MALLOC(uint64 size);

#ifdef USE_MKL_MALLOC
#include <mkl.h>
#define ALIGNED_FREE mkl_free
#else
#include <xmmintrin.h>
#define ALIGNED_FREE _mm_free
#endif

#define ALIGNED_MALLOC_DOUBLES(numDoubles) (double *) ALIGNED_MALLOC((numDoubles)*sizeof(double))
#define ALIGNED_MALLOC_FLOATS(numFloats) (float *) ALIGNED_MALLOC((numFloats)*sizeof(float))
#define ALIGNED_MALLOC_UCHARS(numUchars) (uchar *) ALIGNED_MALLOC((numUchars)*sizeof(uchar))

#endif
