
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
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"
#include "ComputeWAXPBY_ref.hpp"
#include <cstdlib>
#include <stdio.h>

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif


#define BLKSZ 256
#define UNR 6
#define MIN(a,b) ((a)<(b)?(a):(b))

extern "C" {
  void daxpy_(const local_int_t *n, const double *alpha, const double *x,
              const local_int_t *incx, double *y, const local_int_t *incy);
  void dcopy_(const local_int_t *n, const double *x, const local_int_t *incx,
              double *y, const local_int_t *incy);
}

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/
int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  // This line and the next two lines should be removed and your version of ComputeWAXPBY should be used.
  //isOptimized = false;
  //return ComputeWAXPBY_ref(n, alpha, x, beta, y, w);

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  const double * const xv = x.values;
  const double * const yv = y.values;
  double * wv = w.values;
  local_int_t one = 1;

  local_int_t i_, imin, imax;

  if        (alpha==1.0 && beta==0.0)
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
    for (local_int_t i=0; i<n; i++)
      wv[i] = xv[i];
#else
    dcopy_(&n, xv, &one, wv, &one);
#endif

  else if        (alpha==0.0 && beta==1.0)
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
    for (local_int_t i=0; i<n; i++) wv[i] = yv[i];
#else
    dcopy_(&n, yv, &one, wv, &one);
#endif

  else if (alpha==1.0 && &x==&w) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
    for (local_int_t i=0; i<n; i++)
      wv[i] = wv[i] + beta * yv[i];
#else
#if 1
    daxpy_(&n, &beta, yv, &one, wv, &one);
#else
    local_int_t i_;                               
    for (local_int_t i=0; i<n; i+=BLKSZ*UNR) {    
      local_int_t imin = i;                       
      local_int_t imax = MIN(i+BLKSZ*UNR,n);      
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + beta * yv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + beta * yv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + beta * yv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + beta * yv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + beta * yv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + beta * yv[i_];         
    }
#endif
#endif

  } else if (beta==1.0 && &y==&w) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
    for (local_int_t i=0; i<n; i++)
      wv[i] = alpha * xv[i] + wv[i];
#else
#if 1
    daxpy_(&n, &alpha, xv, &one, wv, &one);
#else
    local_int_t i_;                               
    for (local_int_t i=0; i<n; i+=BLKSZ*UNR) {    
      local_int_t imin = i;                       
      local_int_t imax = MIN(i+BLKSZ*UNR,n);      
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + alpha * xv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + alpha * xv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + alpha * xv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + alpha * xv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + alpha * xv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = wv[i_] + alpha * xv[i_];         
    }
#endif
#endif
                                             
    
  } else if (alpha==1.0) {

  local_int_t i_, imin, imax;

#undef UNR
#define UNR 9

#define BLKSZ   256
#define BLKSZ0   0
#define BLKSZ1   (1*BLKSZ)
#define BLKSZ2   (2*BLKSZ)
#define BLKSZ3   (3*BLKSZ)
#define BLKSZ4   (4*BLKSZ)
#define BLKSZ5   (5*BLKSZ)
#define BLKSZ6   (6*BLKSZ)
#define BLKSZ7   (7*BLKSZ)
#define BLKSZ8   (8*BLKSZ)
#define BLKSZ9   (9*BLKSZ)
  
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
  for (local_int_t i=0; i<n; i++) {
    wv[i] = xv[i] + beta * yv[i];
  }
#else

  __builtin_vprefetch((void *)&xv[0], sizeof(double)*BLKSZ*UNR);
  __builtin_vprefetch((void *)&yv[0], sizeof(double)*BLKSZ*UNR);
  __builtin_vprefetch((void *)&wv[0], sizeof(double)*BLKSZ*UNR);

  for (local_int_t i=0; i<n; i+=BLKSZ*UNR) {    
      local_int_t imin = i;                       
      local_int_t imax = MIN(i+BLKSZ*UNR,n);      
      if (imax < n) {
        __builtin_vprefetch((void *)&xv[imax], sizeof(double)*BLKSZ*UNR);
        __builtin_vprefetch((void *)&yv[imax], sizeof(double)*BLKSZ*UNR);
        __builtin_vprefetch((void *)&wv[imax], sizeof(double)*BLKSZ*UNR);
      }
      if (imax < n || imax-imin >= BLKSZ9) {
        #pragma _NEC shortloop                  
        for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++) {
          wv[i_+BLKSZ0] = xv[i_+BLKSZ0] + beta * yv[i_+BLKSZ0];
          wv[i_+BLKSZ1] = xv[i_+BLKSZ1] + beta * yv[i_+BLKSZ1];
          wv[i_+BLKSZ2] = xv[i_+BLKSZ2] + beta * yv[i_+BLKSZ2];
          wv[i_+BLKSZ3] = xv[i_+BLKSZ3] + beta * yv[i_+BLKSZ3];
          wv[i_+BLKSZ4] = xv[i_+BLKSZ4] + beta * yv[i_+BLKSZ4];
          wv[i_+BLKSZ5] = xv[i_+BLKSZ5] + beta * yv[i_+BLKSZ5];
          wv[i_+BLKSZ6] = xv[i_+BLKSZ6] + beta * yv[i_+BLKSZ6];
          wv[i_+BLKSZ7] = xv[i_+BLKSZ7] + beta * yv[i_+BLKSZ7];
          wv[i_+BLKSZ8] = xv[i_+BLKSZ8] + beta * yv[i_+BLKSZ8];
        }
        continue;
      }
      if (imax-imin > BLKSZ6) {
        #pragma _NEC shortloop                  
        for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++) {
          wv[i_+BLKSZ0] = xv[i_+BLKSZ0] + beta * yv[i_+BLKSZ0];
          wv[i_+BLKSZ1] = xv[i_+BLKSZ1] + beta * yv[i_+BLKSZ1];
          wv[i_+BLKSZ2] = xv[i_+BLKSZ2] + beta * yv[i_+BLKSZ2];
          wv[i_+BLKSZ3] = xv[i_+BLKSZ3] + beta * yv[i_+BLKSZ3];
          wv[i_+BLKSZ4] = xv[i_+BLKSZ4] + beta * yv[i_+BLKSZ4];
          wv[i_+BLKSZ5] = xv[i_+BLKSZ5] + beta * yv[i_+BLKSZ5];
        }
        imin += BLKSZ6;

      }
      if (imax-imin > BLKSZ3) {
        #pragma _NEC shortloop                  
        for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++) {
          wv[i_+BLKSZ0] = xv[i_+BLKSZ0] + beta * yv[i_+BLKSZ0];
          wv[i_+BLKSZ1] = xv[i_+BLKSZ1] + beta * yv[i_+BLKSZ1];
          wv[i_+BLKSZ2] = xv[i_+BLKSZ2] + beta * yv[i_+BLKSZ2];
        }
        imin += BLKSZ3;
      }
      if (imax-imin > BLKSZ2) {
        #pragma _NEC shortloop                  
        for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++) {
          wv[i_+BLKSZ0] = xv[i_+BLKSZ0] + beta * yv[i_+BLKSZ0];
          wv[i_+BLKSZ1] = xv[i_+BLKSZ1] + beta * yv[i_+BLKSZ1];
        }
        imin += BLKSZ2;
      }
      if (imax-imin > BLKSZ1) {
        #pragma _NEC shortloop                  
        for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++) {
          wv[i_+BLKSZ0] = xv[i_+BLKSZ0] + beta * yv[i_+BLKSZ0];
        }
        imin += BLKSZ1;
      }
      #pragma _NEC shortloop                  
      for (i_=imin; i_<imax; i_++) {
        wv[i_] = xv[i_] + beta * yv[i_];
      }
  }
#endif

#undef BLKSZ0
#undef BLKSZ1
#undef BLKSZ2
#undef BLKSZ3
#undef BLKSZ4
#undef BLKSZ5
#undef BLKSZ6
#undef BLKSZ7
#undef BLKSZ8
#undef BLKSZ9

  } else if (beta==1.0) {
#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
    for (local_int_t i=0; i<n; i++)
      wv[i] =alpha * xv[i] + yv[i];
#else
    //dcopy_(&n, yv, &one, wv, &one);
    //daxpy_(&n, &alpha, xv, &one, wv, &one);
    local_int_t i_;                               
    for (local_int_t i=0; i<n; i+=BLKSZ*UNR) {    
      local_int_t imin = i;                       
      local_int_t imax = MIN(i+BLKSZ*UNR,n);      
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = yv[i_] + alpha * xv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = yv[i_] + alpha * xv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = yv[i_] + alpha * xv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = yv[i_] + alpha * xv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = yv[i_] + alpha * xv[i_];         
      imin = i_;                                  
      if (i_ >= imax) continue;                   
#pragma _NEC shortloop                  
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] = yv[i_] + alpha * xv[i_];         
    }
#endif

  } else {

#ifndef HPCG_NO_OPENMP
    #pragma omp parallel for
    for (local_int_t i=0; i<n; i++)
      wv[i] = alpha * xv[i] + beta * yv[i];
#else
    for (local_int_t i=0; i<n; i+=BLKSZ*UNR) {
      imin = i;
      imax = MIN(i+BLKSZ*UNR,n);
#pragma _NEC shortloop
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] =alpha * xv[i_] + beta * yv[i_];
      imin = i_;
      if (i_ >= imax) continue;
#pragma _NEC shortloop
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] =alpha * xv[i_] + beta * yv[i_];
      imin = i_;
      if (i_ >= imax) continue;
#pragma _NEC shortloop
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] =alpha * xv[i_] + beta * yv[i_];
      imin = i_;
      if (i_ >= imax) continue;
#pragma _NEC shortloop
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] =alpha * xv[i_] + beta * yv[i_];
      imin = i_;
      if (i_ >= imax) continue;
#pragma _NEC shortloop
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] =alpha * xv[i_] + beta * yv[i_];
      imin = i_;
      if (i_ >= imax) continue;
#pragma _NEC shortloop
      for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++)
        wv[i_] =alpha * xv[i_] + beta * yv[i_];
    }
    //daxpy_(&n, &alpha, xv, &one, wv, &one);
    //daxpy_(&n, &beta, yv, &one, wv, &one);
#endif
    
  }
  return 0;

}

//
// version of WAXPY fused with DotProduct
// only implemented for the case alpha==1, x == w.
// 
int ComputeWAXPBY_DotProd(const local_int_t n, const double alpha, const Vector & x,
                          const double beta, const Vector & y, Vector & w, bool & isOptimized,
                          double &dotp, double &time_allreduce)
{
  const double * const xv = x.values;
  const double * const yv = y.values;
  double * wv = w.values;
  local_int_t one = 1;

  local_int_t i_, imin, imax;

  double dotprod = 0;
  
  assert(alpha==1.0 && &x==&w);

#undef UNR
#define UNR 9

#define BLKSZ   256
#define BLKSZ0   0
#define BLKSZ1   (1*BLKSZ)
#define BLKSZ2   (2*BLKSZ)
#define BLKSZ3   (3*BLKSZ)
#define BLKSZ4   (4*BLKSZ)
#define BLKSZ5   (5*BLKSZ)
#define BLKSZ6   (6*BLKSZ)
#define BLKSZ7   (7*BLKSZ)
#define BLKSZ8   (8*BLKSZ)
#define BLKSZ9   (9*BLKSZ)
  double wtmp0[BLKSZ];
  double wtmp1[BLKSZ];
  double wtmp2[BLKSZ];
  double wtmp3[BLKSZ];
  double wtmp4[BLKSZ];
  double wtmp5[BLKSZ];
  double wtmp6[BLKSZ];
  double wtmp7[BLKSZ];
  double wtmp8[BLKSZ];

  double dptmp[BLKSZ];

#pragma _NEC vreg(wtmp0)
#pragma _NEC vreg(wtmp1)
#pragma _NEC vreg(wtmp2)
#pragma _NEC vreg(wtmp3)
#pragma _NEC vreg(wtmp4)
#pragma _NEC vreg(wtmp5)
#pragma _NEC vreg(wtmp6)
#pragma _NEC vreg(wtmp7)
#pragma _NEC vreg(wtmp8)

  
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
  for (local_int_t i=0; i<n; i++) {
    wv[i] = wv[i] + beta * yv[i];
  }
#else

  __builtin_vprefetch((void *)&wv[0], sizeof(double)*BLKSZ*UNR);
  __builtin_vprefetch((void *)&yv[0], sizeof(double)*BLKSZ*UNR);

  for (int i=0; i<BLKSZ; i++)
      dptmp[i] = 0.0;

  for (local_int_t i=0; i<n; i+=BLKSZ*UNR) {    
      local_int_t imin = i;                       
      local_int_t imax = MIN(i+BLKSZ*UNR,n);      
      if (imax < n) {
        __builtin_vprefetch((void *)&wv[imax], sizeof(double)*BLKSZ*UNR);
        __builtin_vprefetch((void *)&yv[imax], sizeof(double)*BLKSZ*UNR);
      }
      if (imax < n || imax-imin >= BLKSZ9) {
        #pragma _NEC shortloop                  
        for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++) {
          int ib = i_ - imin;
          wtmp0[ib] = wv[i_+BLKSZ0] + beta * yv[i_+BLKSZ0];
          wtmp1[ib] = wv[i_+BLKSZ1] + beta * yv[i_+BLKSZ1];
          wtmp2[ib] = wv[i_+BLKSZ2] + beta * yv[i_+BLKSZ2];
          wtmp3[ib] = wv[i_+BLKSZ3] + beta * yv[i_+BLKSZ3];
          wtmp4[ib] = wv[i_+BLKSZ4] + beta * yv[i_+BLKSZ4];
          wtmp5[ib] = wv[i_+BLKSZ5] + beta * yv[i_+BLKSZ5];
          wtmp6[ib] = wv[i_+BLKSZ6] + beta * yv[i_+BLKSZ6];
          wtmp7[ib] = wv[i_+BLKSZ7] + beta * yv[i_+BLKSZ7];
          wtmp8[ib] = wv[i_+BLKSZ8] + beta * yv[i_+BLKSZ8];
          dptmp[ib] = dptmp[ib]
            + wtmp0[ib]*wtmp0[ib]
            + wtmp1[ib]*wtmp1[ib]
            + wtmp2[ib]*wtmp2[ib]
            + wtmp3[ib]*wtmp3[ib]
            + wtmp4[ib]*wtmp4[ib]
            + wtmp5[ib]*wtmp5[ib]
            + wtmp6[ib]*wtmp6[ib]
            + wtmp7[ib]*wtmp7[ib]
            + wtmp8[ib]*wtmp8[ib];

            wv[i_+BLKSZ0] = wtmp0[ib]; 
            wv[i_+BLKSZ1] = wtmp1[ib];
            wv[i_+BLKSZ2] = wtmp2[ib];
            wv[i_+BLKSZ3] = wtmp3[ib];
            wv[i_+BLKSZ4] = wtmp4[ib];
            wv[i_+BLKSZ5] = wtmp5[ib];
            wv[i_+BLKSZ6] = wtmp6[ib];
            wv[i_+BLKSZ7] = wtmp7[ib];
            wv[i_+BLKSZ8] = wtmp8[ib];
        }
        continue;
      }
      if (imax-imin > BLKSZ6) {
        #pragma _NEC shortloop                  
        for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++) {
          int ib = i_ - imin;
          wtmp0[ib] = wv[i_+BLKSZ0] + beta * yv[i_+BLKSZ0];
          wtmp1[ib] = wv[i_+BLKSZ1] + beta * yv[i_+BLKSZ1];
          wtmp2[ib] = wv[i_+BLKSZ2] + beta * yv[i_+BLKSZ2];
          wtmp3[ib] = wv[i_+BLKSZ3] + beta * yv[i_+BLKSZ3];
          wtmp4[ib] = wv[i_+BLKSZ4] + beta * yv[i_+BLKSZ4];
          wtmp5[ib] = wv[i_+BLKSZ5] + beta * yv[i_+BLKSZ5];
          dptmp[ib] = dptmp[ib]
            + wtmp0[ib]*wtmp0[ib]
            + wtmp1[ib]*wtmp1[ib]
            + wtmp2[ib]*wtmp2[ib]
            + wtmp3[ib]*wtmp3[ib]
            + wtmp4[ib]*wtmp4[ib]
            + wtmp5[ib]*wtmp5[ib];
          wv[i_+BLKSZ0] = wtmp0[ib]; 
          wv[i_+BLKSZ1] = wtmp1[ib];
          wv[i_+BLKSZ2] = wtmp2[ib];
          wv[i_+BLKSZ3] = wtmp3[ib];
          wv[i_+BLKSZ4] = wtmp4[ib];
          wv[i_+BLKSZ5] = wtmp5[ib];
        }
        imin += BLKSZ6;

      }
      if (imax-imin > BLKSZ3) {
        #pragma _NEC shortloop                  
        for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++) {
          int ib = i_ - imin;
          wtmp0[ib] = wv[i_+BLKSZ0] + beta * yv[i_+BLKSZ0];
          wtmp1[ib] = wv[i_+BLKSZ1] + beta * yv[i_+BLKSZ1];
          wtmp2[ib] = wv[i_+BLKSZ2] + beta * yv[i_+BLKSZ2];
          dptmp[ib] = dptmp[ib]
            + wtmp0[ib]*wtmp0[ib]
            + wtmp1[ib]*wtmp1[ib]
            + wtmp2[ib]*wtmp2[ib];
          wv[i_+BLKSZ0] = wtmp0[ib]; 
          wv[i_+BLKSZ1] = wtmp1[ib];
          wv[i_+BLKSZ2] = wtmp2[ib];
        }
        imin += BLKSZ3;
      }
      if (imax-imin > BLKSZ2) {
        #pragma _NEC shortloop                  
        for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++) {
          int ib = i_ - imin;
          wtmp0[ib] = wv[i_+BLKSZ0] + beta * yv[i_+BLKSZ0];
          wtmp1[ib] = wv[i_+BLKSZ1] + beta * yv[i_+BLKSZ1];
          dptmp[ib] = dptmp[ib]
            + wtmp0[ib]*wtmp0[ib]
            + wtmp1[ib]*wtmp1[ib];
          wv[i_+BLKSZ0] = wtmp0[ib]; 
          wv[i_+BLKSZ1] = wtmp1[ib];
        }
        imin += BLKSZ2;
      }
      if (imax-imin > BLKSZ1) {
        #pragma _NEC shortloop                  
        for (i_=imin; i_<MIN(imin+BLKSZ,imax); i_++) {
          int ib = i_ - imin;
          wtmp0[ib] = wv[i_+BLKSZ0] + beta * yv[i_+BLKSZ0];
          dptmp[ib] = dptmp[ib]
            + wtmp0[ib]*wtmp0[ib];
          wv[i_+BLKSZ0] = wtmp0[ib]; 
        }
        imin += BLKSZ1;
      }
      #pragma _NEC shortloop                  
      for (i_=imin; i_<imax; i_++) {
        int ib = i_ - imin;
        wtmp0[ib] = wv[i_] + beta * yv[i_];
        dptmp[ib] = dptmp[ib] + wtmp0[ib]*wtmp0[ib];
        wv[i_] = wtmp0[ib]; 
      }
  }

    #pragma _NEC shortloop
    for (int i=0; i<BLKSZ; i++)
      dotprod += dptmp[i];
#endif

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  // workaround: when using dotprod in the MPI_Allreduce
  // the reductions which aggregate it won't vectorize
  double my_result = dotprod;
  MPI_Allreduce(&my_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  dotp = global_result;
  time_allreduce += mytimer() - t0;
#else
  dotp = dotprod;
#endif

  return 0;

}
