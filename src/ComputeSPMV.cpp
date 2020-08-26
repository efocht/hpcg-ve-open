
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
 @file ComputeSPMV.cpp

 HPCG routine
 */
#include <algorithm>
#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include "OptimizeProblem.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#include "mytimer.hpp"
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif
#include <cassert>
#include <iostream>

#define INTR

using namespace std;

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y)
{
  OPT                   *opt = (OPT*)(A.optimizationData);
  const ELL             *ell = opt->ell;
  const double            *a = ell->a;
  const local_int_t      *ja = ell->ja;
  const local_int_t      lda = ell->lda;
  const local_int_t        m = ell->m;
  const local_int_t       mL = ell->mL;
  const local_int_t       mU = ell->mU;
  const local_int_t        n = ell->n;
  const double         *diag = opt->diag;
  local_int_t mm;

  const HALO         *halo = opt->halo;
  const double         *ah = halo->ah;
  const local_int_t   *jah = halo->jah;
  const local_int_t *hrows = halo->rows;
  const local_int_t   ldah = halo->ldah;
  const local_int_t     mh = halo->mh;
  const local_int_t    nah = halo->nah;
  double               *vh = halo->v;

  mm = mL+mU;

#ifndef HPCG_NO_MPI
  MPI_Request *requests = NULL;
  int num_requests;
  ExchangeHalo_nowait(A, x, 0, &requests, num_requests);
  //ExchangeHalo(A, x, 0);
#endif

  ell_b0_spmv_probe(a, diag, lda, n, mm, ja, x.values, y.values);

#ifndef HPCG_NO_MPI
  ExchangeHalo_wait(0, &requests, num_requests);
#endif

  // halo matrix multiplied with (remote) halo X elements
#ifndef INTR
  dwmve0_spmv(ah, &ldah, &nah, &mh, jah, &x.values[n], vh);
#else
  spmv_intr_regs(ah, ldah, nah, mh, jah, &x.values[n], vh);
#endif

#pragma _NEC ivdep
  for (local_int_t ih = 0; ih < nah; ih++) {
    local_int_t i = hrows[ih];
    y.values[i] += vh[ih];
  }
  return 0;
}

int ComputeSPMV_L( const SparseMatrix & A, Vector & x, Vector & y)
{

  OPT                   *opt = (OPT*)(A.optimizationData);
  const ELL             *ell = opt->ell;
  const double            *a = ell->a;
  const local_int_t      *ja = ell->ja;
  const local_int_t      lda = ell->lda;
  const local_int_t        m = ell->m;
  const local_int_t       mL = ell->mL;
  const local_int_t       mU = ell->mU;
  const local_int_t        n = ell->n;
  const double         *diag = opt->diag;

  const HALO       *halo = opt->halo;
  const double       *ah = halo->ah;
  const local_int_t *jah = halo->jah;
  const local_int_t *hrows = halo->rows;
  const local_int_t ldah = halo->ldah;
  const local_int_t   mh = halo->mh;
  const local_int_t  nah = halo->nah;
  double             *vh = halo->v;

#ifndef HPCG_NO_MPI
  MPI_Request *requests = NULL;
  int num_requests;
  ExchangeHalo_nowait(A, x, 0, &requests, num_requests);
  //ExchangeHalo(A, x, 0);
#endif

  ell_b0_spmv_add_probe(a, NULL, lda, n, mL, ja, x.values, y.values, opt->work2);

#ifndef HPCG_NO_MPI
  ExchangeHalo_wait(0, &requests, num_requests);
#endif

// halo matrix multiplied with (remote) halo X elements
#ifndef INTR
  dwmve0_spmv(ah, &ldah, &nah, &mh, jah, &x.values[n], vh);
#else
  spmv_intr_regs(ah, ldah, nah, mh, jah, &x.values[n], vh);
#endif

#pragma _NEC ivdep
  for (local_int_t ih = 0; ih < nah; ih++) {
    local_int_t i = hrows[ih];
    y.values[i] += vh[ih];
  }
  return 0;
}

//
// version which is fused with the dot product
//
// replaces
//    ComputeSPMV(A, p, Ap);  // Ap = A*p
//    ComputeDotProduct(nrow, p, Ap, pAp, t4, A.isDotProductOptimized, false, nullptr); // alpha = p'*Ap
//
int ComputeSPMV_DotProd( const SparseMatrix & A, Vector & x, Vector & y,
                         double &alpha, double &time_allreduce)
{

  OPT                   *opt = (OPT*)(A.optimizationData);
  const ELL             *ell = opt->ell;
  const double            *a = ell->a;
  const local_int_t      *ja = ell->ja;
  const local_int_t      lda = ell->lda;
  const local_int_t        m = ell->m;
  const local_int_t       mL = ell->mL;
  const local_int_t       mU = ell->mU;
  const local_int_t        n = ell->n;
  const double         *diag = opt->diag;
  local_int_t mm;

  const HALO       *halo = opt->halo;
  const double       *ah = halo->ah;
  const local_int_t *jah = halo->jah;
  const local_int_t *hrows = halo->rows;
  const local_int_t ldah = halo->ldah;
  const local_int_t   mh = halo->mh;
  const local_int_t  nah = halo->nah;
  double             *vh = halo->v;
  
  mm = mL+mU;

#ifndef HPCG_NO_MPI
  MPI_Request *requests = NULL;
  int num_requests;
  ExchangeHalo_nowait(A, x, 0, &requests, num_requests);
  //ExchangeHalo(A, x, 0);
#endif

  // blocking for increasing cache reuse

  double dotprod = 0;

  ell_b0_spmv_fusedot_probe(a, diag, lda, n, mm, ja, x.values, y.values, &dotprod);

#ifndef HPCG_NO_MPI
  ExchangeHalo_wait(0, &requests, num_requests);
#endif
  // halo matrix multiplied with (remote) halo X elements
#ifndef INTR
  dwmve0_spmv(ah, &ldah, &nah, &mh, jah, &x.values[n], vh);
#else
  spmv_intr_regs(ah, ldah, nah, mh, jah, &x.values[n], vh);
#endif

#pragma _NEC ivdep
  for (local_int_t ih = 0; ih < nah; ih++) {
    local_int_t i = hrows[ih];
    y.values[i] += vh[ih];
    dotprod += x.values[i] * vh[ih];
  }

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  // workaround: when using dotprod in the MPI_Allreduce
  // the reductions which aggregate it won't vectorize
  double my_result = dotprod;
  MPI_Allreduce(&my_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  alpha = global_result;
  time_allreduce += mytimer() - t0;
#else
  alpha = dotprod;
#endif

  return 0;
}

