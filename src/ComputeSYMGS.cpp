
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
// Erich Focht : Version optimized for SX-Aurora
//
// ***************************************************
//@HEADER

/*!
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"
#include "OptimizeProblem.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#include <mpi.h>
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#include "omp_barrier.h"
#endif

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
using namespace std;

int ComputeSYMGS_mpi_only( const SparseMatrix & A, const Vector & r, Vector & x, int is_first) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values
   
  OPT* opt         = (OPT*)(A.optimizationData);
  const ELL         *ell = opt->ell;
  const double        *a = ell->a;
  const local_int_t  *ja = ell->ja;
  const local_int_t  lda = ell->lda;
  const local_int_t    m = ell->m;
  const local_int_t   mL = ell->mL;
  const local_int_t   mU = ell->mU;
  const local_int_t    n = ell->n;
  const double     *diag = opt->diag;
  double          *idiag = opt->idiag;
  double          *work1 = opt->work1;
  double          *work2 = opt->work2;
  const local_int_t *icptr    = opt->icptr;
  const local_int_t  maxcolor = opt->maxcolor;
  const local_int_t *color_mL = opt->color_mL;
  const local_int_t *color_mU = opt->color_mU;

  const HALO       *halo = opt->halo;
  const double       *ah = halo->ah;
  const local_int_t *jah = halo->jah;
  const local_int_t *hcptr = halo->hcptr;
  const local_int_t *hrows = halo->rows;
  const local_int_t ldah = halo->ldah;
  const local_int_t   mh = halo->mh;
  const local_int_t  nah = halo->nah;
  double             *vh = halo->v;

  double dummy;
  
  const double * const rv = r.values;
  double * const xv = x.values;
  local_int_t irs, ire, nn, ics, ice;

  MPI_Request *requests = NULL;
  int num_requests;
  ExchangeHalo_nowait(A, x, is_first, &requests, num_requests);
  //ExchangeHalo(A, x, is_first);

  if (is_first) {

    //////////////////
    // Forward sweep
    //////////////////

    // initial guess x_old is zero
    //
    // # (L+D+U)x = b
    // # 1. (L+D)*(x_new) = b
    //     #1.3 TRSV(L+D, b, x_new) #Similar to GS

    for (local_int_t i = 0; i < n; i++)
      work1[i] = rv[i];

    ell_col_b0_trsv(0, maxcolor, icptr, color_mL, a, idiag, lda, ja, xv, work1);

    for (local_int_t i=0; i<n; i++) {
      //work1[i] = diag[i] * xv[i];  // this is already done as xv[i] = work1[i] * idiag[i]
      xv[i] = 0.0;
    }

    ///////////////////
    // Backward sweep
    ///////////////////

    // # 2.(U+D)*x_new_new = b - L*x_new
    //     #2.1 x_new = b-L*x_new = D*x_new
    //     #2.2 TRSV(U+D, x_new, x_new_new)

    for (local_int_t ic = maxcolor - 1; ic >= 0; ic--){
      irs = icptr[ic];
      ire = icptr[ic+1];

      for (local_int_t i = irs; i < ire; i++)
        work2[i] = work1[i];   // work1 is being modified inside the sweep,
                               // we need work2 in SPMV

      // upper matrix GS sweep 
      ell_b0_trsv_step(irs, ire, &a[lda * mL], idiag, lda,
                       color_mU[ic], &ja[lda * mL], xv, work1);

      // halo contribution is zero because initial guess is zero
    }
    
  } else { // ! is_first

    //////////////////
    // Forward sweep
    //////////////////

    //# (L+D+U)x = b
    //# 1. (L+D)*(x_new) = b - U*x_old
    //    #1.1 U*x_old = tmp1
    //    #1.2 b-tmp1 = tmp
    //    #1.3 TRSV(L+D, tmp, x_new) #Similar to GS
  
    // #1.1 U*xv  -> work1

    ell_b0_spmv_probe(&a[lda * mL], NULL, lda, n, mU, &ja[lda * mL], xv, work1);

    ExchangeHalo_wait(0, &requests, num_requests);

    // vh <- Ah * xv[n:]
    dwmve0_spmv(ah, &ldah, &nah, &mh, jah, &xv[n], vh);
    // halo matrix multiplied with (remote) halo X elements
    #pragma _NEC ivdep
    for (local_int_t ih = 0; ih < nah; ih++) {
      local_int_t i = hrows[ih];
      work1[i] += vh[ih];
    }
    
    // #1.2 b-tmp1 = tmp
    for (local_int_t i = 0; i < n; i++) {
      work2[i] = rv[i] - work1[i];
      xv[i] = 0.0;
    }
  
    // #1.3 TRSV(L+D, tmp, x_new) #Similar to GS
    for(local_int_t ic = 0; ic < maxcolor; ic++){
      irs = icptr[ic];
      ire = icptr[ic+1];
      // lower matrix TRSV sweep 
      ell_b0_trsv_step(irs, ire, a, idiag, lda, color_mL[ic], ja, xv, work2);
      //
      for (local_int_t i = irs; i < ire; i++) {
        work2[i] += work1[i];
      }
    }

    ///////////////////
    // Backward sweep
    ///////////////////

    // # 2.(U+D)*x_new_new = b - L*x_new
    //     #2.1 x_new = b-L*x_new = D*x_new + tmp1
    //     #2.2 TRSV(U+D, x_new, x_new_new)

    // 2.1 x_new = b-L*x_new = D*x_new + tmp1
    for (local_int_t i = 0; i < n; i++) {
      //work2[i] = work1[i] + diag[i]*xv[i];   // but xv[i] = work2[i] * idiag[i], so...
      //work2[i] += work1[i];  // done in forward sweep
      xv[i] = 0.0;
    }

    // 2.2 TRSV(U+D, x_new, x_new_new)
    for (local_int_t ic = maxcolor - 1; ic >= 0; ic--){
      irs = icptr[ic];
      ire = icptr[ic+1];

      // upper matrix GS sweep 
      ell_b0_trsv_step(irs, ire, &a[lda * mL], idiag, lda,
                       color_mU[ic], &ja[lda * mL], xv, work2);

      #pragma _NEC ivdep
      for (local_int_t ih = hcptr[ic]; ih < hcptr[ic + 1]; ih++) {
        local_int_t i = hrows[ih];
        xv[i] -= vh[ih] * idiag[i];
      }
    }

  }

  return 0;

}



/*!
  Routine to one step of symmetrix Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in]  A the known system matrix
  @param[in]  x the input vector
  @param[out] y On exit contains the result of one symmetric GS sweep with x as the RHS.
  @param[in]  is_first Flag signalling that this is the first call of SYMGS

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x, int is_first)
{
#ifndef HPCG_NO_OPENMP
  return ComputeSYMGS_omp(A, r, x, is_first);
#else
  return ComputeSYMGS_mpi_only(A, r, x, is_first);
#endif
}
