
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif

#include <iostream>

extern "C" {
  extern int ftrace_region_begin(const char *id);
  extern int ftrace_region_end(const char *id);
}

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
                      double & result, double & time_allreduce, bool & isOptimized, const bool async, void *asyncreq)
{

  // This line and the next two lines should be removed and your version of ComputeDotProduct should be used.
  isOptimized = true;
  double * xv = x.values;
  double * yv = y.values;
  double local_result = 0.0;

#ifdef HPCG_NO_OPENMP  
//EF//#ifndef HPCG_NO_OPENMP
//EF//#pragma omp parallel default(shared) reduction(+:local_result)
//EF//  {
//EF//#endif
  
  int inc = 1;
  local_int_t imin, imax, i_;

#define BLKSZ 256
#define BLKSZ2   (2*BLKSZ)
#define BLKSZ3   (3*BLKSZ)
#define BLKSZ4   (4*BLKSZ)
#define BLKSZ5   (5*BLKSZ)
#define BLKSZ6   (6*BLKSZ)
#define BLKSZ7   (7*BLKSZ)
#define BLKSZ8   (8*BLKSZ)
#define BLKSZ9   (9*BLKSZ)
#define BLKSZ10  (10*BLKSZ)
#define BLKSZ11  (11*BLKSZ)
#define BLKSZ12  (12*BLKSZ)
#define BLKSZ13  (13*BLKSZ)
#define BLKSZ14  (14*BLKSZ)
#define BLKSZ15  (15*BLKSZ)
#define BLKSZ16  (16*BLKSZ)
#define UNR 16
#define MIN(a,b) ((a)<(b)?(a):(b))

  double lsum0[BLKSZ];
  double lsum1[BLKSZ];
  double lsum2[BLKSZ];
  double lsum3[BLKSZ];
  double lsum4[BLKSZ];
  double lsum5[BLKSZ];
  double lsum6[BLKSZ];
  double lsum7[BLKSZ];
  double lsum8[BLKSZ];
  double lsum9[BLKSZ];
  double lsum10[BLKSZ];
  double lsum11[BLKSZ];
  double lsum12[BLKSZ];
  double lsum13[BLKSZ];
  double lsum14[BLKSZ];
  double lsum15[BLKSZ];
#pragma _NEC vreg(lsum0)
#pragma _NEC vreg(lsum1)
#pragma _NEC vreg(lsum2)
#pragma _NEC vreg(lsum3)
#pragma _NEC vreg(lsum4)
#pragma _NEC vreg(lsum5)
#pragma _NEC vreg(lsum6)
#pragma _NEC vreg(lsum7)
#pragma _NEC vreg(lsum8)
#pragma _NEC vreg(lsum9)
#pragma _NEC vreg(lsum10)
#pragma _NEC vreg(lsum11)
#pragma _NEC vreg(lsum12)
#pragma _NEC vreg(lsum13)
#pragma _NEC vreg(lsum14)
#pragma _NEC vreg(lsum15)

  if (xv != yv) {
#ifdef FTRACE
    ftrace_region_begin("DotProduct_XY");
#endif
    //local_result = ddot_(&n, xv, &inc, yv, &inc);

    __builtin_vprefetch((void *)&xv[0], sizeof(double)*BLKSZ*UNR);
    __builtin_vprefetch((void *)&yv[0], sizeof(double)*BLKSZ*UNR);
#pragma _NEC shortloop
    for (int i=0; i<BLKSZ; i++) {
      lsum0[i] = 0.0;
      lsum1[i] = 0.0;
      lsum2[i] = 0.0;
      lsum3[i] = 0.0;
      lsum4[i] = 0.0;
      lsum5[i] = 0.0;
      lsum6[i] = 0.0;
      lsum7[i] = 0.0;
      lsum8[i] = 0.0;
      lsum9[i] = 0.0;
      lsum10[i] = 0.0;
      lsum11[i] = 0.0;
      lsum12[i] = 0.0;
      lsum13[i] = 0.0;
      lsum14[i] = 0.0;
      lsum15[i] = 0.0;
    }
#pragma omp for schedule(static)
    for (local_int_t i=0; i<n; i+=BLKSZ*UNR) {
      imin = i;
      imax = MIN(i+BLKSZ*UNR,n);
      if (imax < n) {
        __builtin_vprefetch((void *)&xv[imax], sizeof(double)*BLKSZ*UNR);
        __builtin_vprefetch((void *)&yv[imax], sizeof(double)*BLKSZ*UNR);
      }
      if (imax < n || imax-imin >= BLKSZ16) {
#pragma _NEC shortloop
        for (i_=0; i_<BLKSZ; i_++) {
          lsum0[i_]  += xv[i_+imin]          * yv[i_+imin];
          lsum1[i_]  += xv[i_+imin+BLKSZ]    * yv[i_+imin+BLKSZ];
          lsum2[i_]  += xv[i_+imin+BLKSZ2]   * yv[i_+imin+BLKSZ2];
          lsum3[i_]  += xv[i_+imin+BLKSZ3]   * yv[i_+imin+BLKSZ3];
          lsum4[i_]  += xv[i_+imin+BLKSZ4]   * yv[i_+imin+BLKSZ4];
          lsum5[i_]  += xv[i_+imin+BLKSZ5]   * yv[i_+imin+BLKSZ5];
          lsum6[i_]  += xv[i_+imin+BLKSZ6]   * yv[i_+imin+BLKSZ6];
          lsum7[i_]  += xv[i_+imin+BLKSZ7]   * yv[i_+imin+BLKSZ7];
          lsum8[i_]  += xv[i_+imin+BLKSZ8]   * yv[i_+imin+BLKSZ8];
          lsum9[i_]  += xv[i_+imin+BLKSZ9]   * yv[i_+imin+BLKSZ9];
          lsum10[i_] += xv[i_+imin+BLKSZ10]  * yv[i_+imin+BLKSZ10];
          lsum11[i_] += xv[i_+imin+BLKSZ11]  * yv[i_+imin+BLKSZ11];
          lsum12[i_] += xv[i_+imin+BLKSZ12]  * yv[i_+imin+BLKSZ12];
          lsum13[i_] += xv[i_+imin+BLKSZ13]  * yv[i_+imin+BLKSZ13];
          lsum14[i_] += xv[i_+imin+BLKSZ14]  * yv[i_+imin+BLKSZ14];
          lsum15[i_] += xv[i_+imin+BLKSZ15]  * yv[i_+imin+BLKSZ15];
        }
        continue;
      }
      if (imax-imin > BLKSZ8) {
#pragma _NEC shortloop
        for (i_=0; i_<BLKSZ; i_++) {
          lsum0[i_] += xv[i_+imin]         * yv[i_+imin];
          lsum1[i_] += xv[i_+imin+BLKSZ]   * yv[i_+imin+BLKSZ];
          lsum2[i_] += xv[i_+imin+BLKSZ2]  * yv[i_+imin+BLKSZ2];
          lsum3[i_] += xv[i_+imin+BLKSZ3]  * yv[i_+imin+BLKSZ3];
          lsum4[i_] += xv[i_+imin+BLKSZ4]  * yv[i_+imin+BLKSZ4];
          lsum5[i_]  += xv[i_+imin+BLKSZ5]   * yv[i_+imin+BLKSZ5];
          lsum6[i_]  += xv[i_+imin+BLKSZ6]   * yv[i_+imin+BLKSZ6];
          lsum7[i_]  += xv[i_+imin+BLKSZ7]   * yv[i_+imin+BLKSZ7];
        }
        imin += BLKSZ8;
      }
      if (imax-imin > BLKSZ4) {
#pragma _NEC shortloop
        for (i_=0; i_<BLKSZ; i_++) {
          lsum0[i_] += xv[i_+imin]         * yv[i_+imin];
          lsum1[i_] += xv[i_+imin+BLKSZ]   * yv[i_+imin+BLKSZ];
          lsum2[i_] += xv[i_+imin+BLKSZ2]  * yv[i_+imin+BLKSZ2];
          lsum3[i_] += xv[i_+imin+BLKSZ3]  * yv[i_+imin+BLKSZ3];
        }
        imin += BLKSZ4;
      }
      if (imax-imin > BLKSZ2) {
#pragma _NEC shortloop
        for (i_=0; i_<BLKSZ; i_++) {
          lsum0[i_] += xv[i_+imin]         * yv[i_+imin];
          lsum1[i_] += xv[i_+imin+BLKSZ]   * yv[i_+imin+BLKSZ];
        }
        imin += BLKSZ2;
      }
      if (imax-imin > BLKSZ) {
#pragma _NEC shortloop
        for (i_=0; i_<BLKSZ; i_++) {
          lsum0[i_] += xv[i_+imin]         * yv[i_+imin];
        }
        imin += BLKSZ;
      }
#pragma _NEC shortloop
      for (i_=imin; i_<imax; i_++)
        lsum0[i_-imin] += xv[i_] * yv[i_];
    }
    double dsum = 0.0;
#pragma _NEC shortloop
    for (int i=0; i<BLKSZ; i++) {
      dsum += lsum0[i] + lsum1[i] + lsum2[i] + lsum3[i] + lsum4[i] + lsum5[i] +
        lsum6[i] + lsum7[i] + lsum8[i] + lsum9[i] + lsum10[i] + lsum11[i] +
        lsum12[i] + lsum13[i] + lsum14[i] + lsum15[i];
    }
    // leave this here for OpenMP
    local_result += dsum;
#ifdef FTRACE    
    ftrace_region_end("DotProduct_XY");
#endif

  } else {  // xv == yv

#ifdef FTRACE    
    ftrace_region_begin("DotProduct_XX");
#endif
    __builtin_vprefetch((void *)&xv[0], sizeof(double)*BLKSZ*UNR);
#pragma _NEC shortloop
    for (int i=0; i<BLKSZ; i++) {
      lsum0[i] = 0.0;
      lsum1[i] = 0.0;
      lsum2[i] = 0.0;
      lsum3[i] = 0.0;
      lsum4[i] = 0.0;
      lsum5[i] = 0.0;
      lsum6[i] = 0.0;
      lsum7[i] = 0.0;
      lsum8[i] = 0.0;
      lsum9[i] = 0.0;
      lsum10[i] = 0.0;
      lsum11[i] = 0.0;
      lsum12[i] = 0.0;
      lsum13[i] = 0.0;
      lsum14[i] = 0.0;
      lsum15[i] = 0.0;
    }
#pragma omp for schedule(static)
    for (local_int_t i=0; i<n; i+=BLKSZ*UNR) {
      imin = i;
      imax = MIN(i+BLKSZ*UNR,n);
      if (imax < n) {
        __builtin_vprefetch((void *)&xv[imax], sizeof(double)*BLKSZ*UNR);
      }
      if (imax < n || imax-imin >= BLKSZ16) {
#pragma _NEC shortloop
        for (i_=0; i_<BLKSZ; i_++) {
          lsum0[i_]  += xv[i_+imin]          * xv[i_+imin];
          lsum1[i_]  += xv[i_+imin+BLKSZ]    * xv[i_+imin+BLKSZ];
          lsum2[i_]  += xv[i_+imin+BLKSZ2]   * xv[i_+imin+BLKSZ2];
          lsum3[i_]  += xv[i_+imin+BLKSZ3]   * xv[i_+imin+BLKSZ3];
          lsum4[i_]  += xv[i_+imin+BLKSZ4]   * xv[i_+imin+BLKSZ4];
          lsum5[i_]  += xv[i_+imin+BLKSZ5]   * xv[i_+imin+BLKSZ5];
          lsum6[i_]  += xv[i_+imin+BLKSZ6]   * xv[i_+imin+BLKSZ6];
          lsum7[i_]  += xv[i_+imin+BLKSZ7]   * xv[i_+imin+BLKSZ7];
          lsum8[i_]  += xv[i_+imin+BLKSZ8]   * xv[i_+imin+BLKSZ8];
          lsum9[i_]  += xv[i_+imin+BLKSZ9]   * xv[i_+imin+BLKSZ9];
          lsum10[i_] += xv[i_+imin+BLKSZ10]  * xv[i_+imin+BLKSZ10];
          lsum11[i_] += xv[i_+imin+BLKSZ11]  * xv[i_+imin+BLKSZ11];
          lsum12[i_] += xv[i_+imin+BLKSZ12]  * xv[i_+imin+BLKSZ12];
          lsum13[i_] += xv[i_+imin+BLKSZ13]  * xv[i_+imin+BLKSZ13];
          lsum14[i_] += xv[i_+imin+BLKSZ14]  * xv[i_+imin+BLKSZ14];
          lsum15[i_] += xv[i_+imin+BLKSZ15]  * xv[i_+imin+BLKSZ15];
        }
        continue;
      }
      if (imax-imin > BLKSZ8) {
#pragma _NEC shortloop
        for (i_=0; i_<BLKSZ; i_++) {
          lsum0[i_] += xv[i_+imin]          * xv[i_+imin];
          lsum1[i_] += xv[i_+imin+BLKSZ]    * xv[i_+imin+BLKSZ];
          lsum2[i_] += xv[i_+imin+BLKSZ2]   * xv[i_+imin+BLKSZ2];
          lsum3[i_] += xv[i_+imin+BLKSZ3]   * xv[i_+imin+BLKSZ3];
          lsum4[i_] += xv[i_+imin+BLKSZ4]   * xv[i_+imin+BLKSZ4];
          lsum5[i_]  += xv[i_+imin+BLKSZ5]  * xv[i_+imin+BLKSZ5];
          lsum6[i_]  += xv[i_+imin+BLKSZ6]  * xv[i_+imin+BLKSZ6];
          lsum7[i_]  += xv[i_+imin+BLKSZ7]  * xv[i_+imin+BLKSZ7];
        }
        imin += BLKSZ8;
      }
      if (imax-imin > BLKSZ4) {
#pragma _NEC shortloop
        for (i_=0; i_<BLKSZ; i_++) {
          lsum0[i_] += xv[i_+imin]         * xv[i_+imin];
          lsum1[i_] += xv[i_+imin+BLKSZ]   * xv[i_+imin+BLKSZ];
          lsum2[i_] += xv[i_+imin+BLKSZ2]  * xv[i_+imin+BLKSZ2];
          lsum3[i_] += xv[i_+imin+BLKSZ3]  * xv[i_+imin+BLKSZ3];
        }
        imin += BLKSZ4;
      }
      if (imax-imin > BLKSZ2) {
#pragma _NEC shortloop
        for (i_=0; i_<BLKSZ; i_++) {
          lsum0[i_] += xv[i_+imin]         * xv[i_+imin];
          lsum1[i_] += xv[i_+imin+BLKSZ]   * xv[i_+imin+BLKSZ];
        }
        imin += BLKSZ2;
      }
      if (imax-imin > BLKSZ) {
#pragma _NEC shortloop
        for (i_=0; i_<BLKSZ; i_++) {
          lsum0[i_] += xv[i_+imin]         * xv[i_+imin];
        }
        imin += BLKSZ;
      }
#pragma _NEC shortloop
      for (i_=imin; i_<imax; i_++)
        lsum0[i_-imin] += xv[i_] * xv[i_];
    }
    double dsum = 0.0;
#pragma _NEC shortloop
    for (int i=0; i<BLKSZ; i++) {
      local_result += lsum0[i] + lsum1[i] + lsum2[i] + lsum3[i] + lsum4[i] + lsum5[i] +
        lsum6[i] + lsum7[i] + lsum8[i] + lsum9[i] + lsum10[i] + lsum11[i] +
        lsum12[i] + lsum13[i] + lsum14[i] + lsum15[i];
    }
    // leave this here for openmp
    local_result += dsum;
  }
#ifdef FTRACE
  ftrace_region_end("DotProduct_XX");
#endif

//EF//#ifndef HPCG_NO_OPENMP
//EF//  }
//EF//#endif

#else // ! HPCG_NO_OPENMP

  if (xv != yv) {
#pragma omp parallel for reduction(+:local_result)  
    for (local_int_t i=0; i<n; i++) {
      local_result += xv[i] * yv[i];
    }
  } else {
#pragma omp parallel for reduction(+:local_result)  
    for (local_int_t i=0; i<n; i++) {
      local_result += xv[i] * xv[i];
    }
  }

#endif

#ifndef HPCG_NO_MPI

#ifdef FTRACE
  ftrace_region_begin("DotProduct_allreduce");
#endif
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  // workaround: when using local_result in the MPI_Allreduce
  // the reductions which aggregate it won't vectorize
  double my_result = local_result;
  if (async) {
    //MPI_Iallreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
    //               MPI_COMM_WORLD, (MPI_Request *)asyncreq);
    //MPI_Status status;
    //if ( MPI_Wait((MPI_Request *)asyncreq, &status) ) {
    //  std::exit(-1); // TODO: have better error exit
    //}
    //
    // allreduce happens outside!
    //
    global_result = local_result;
  } else
    MPI_Allreduce(&my_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#ifdef FTRACE
  ftrace_region_end("DotProduct_allreduce");
#endif
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}
