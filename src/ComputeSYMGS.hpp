
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

#ifndef COMPUTESYMGS_HPP
#define COMPUTESYMGS_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "ComputeSPMV.hpp"
#include "vel_hpcg_kernels.hpp"

extern "C" {
  void dwmve0_gs(const double* a, const local_int_t* lda, const local_int_t* n,
                 const local_int_t* m, const local_int_t* ja, const double* x, double* y);
  void dwmve0_spmv(const double* a, const local_int_t* lda, const local_int_t* n,
                   const local_int_t* m, const local_int_t* ja, const double* x, double* y);
}

//
// Naive implementation of Gauss Seidel substitution step, row-wise
//
static void dwmve0_gs_rowwise(const double* const a, const int *lda_,
                              const int *n_, const int *m_, const int* const ja,
                              const double* const vi, double* const vo)
{
  int n = *n_;
  int m = *m_;
  int lda = *lda_;

  for (int i = 0; i < n; i++) {
    double sum = vo[i];
    for (int j = 0; j < m; j++) {
      sum -= a[i + lda * j] * vi[ja[i + lda * j]];
    }
    vo[i] = sum;
  }
}

//
// Simple implementation of column-wise Gauss-Seidel substitution step
//
static inline void
dwmve0_gs_colwise(const local_int_t ics, const local_int_t ice, const double *a,
                  const double *idiag, const local_int_t lda, const local_int_t m,
                  const local_int_t *ja, double *xv, double *work)
{
  for (local_int_t j = 0; j < m; j++)
    for (local_int_t i = ics; i < ice; i++)
      work[i] -= a[i + lda * j] * xv[ja[i + lda * j]];
  for (local_int_t i = ics; i < ice; i++)
    xv[i] = work[i] * idiag[i];
}

//
// Triangular solve step for one color of a column-wise stored sparse matrix in
// ELLPACK format with rows reordered by increasing colors.
//
static inline void
ell_b0_trsv_step(const local_int_t irs, const local_int_t ire, const double *a,
                 const double *idiag, const local_int_t lda, const local_int_t m,
                 const local_int_t *ja, double *xv, double *work)
{
  local_int_t nn = ire - irs;

  if(nn >= VLEN) {
    /* Tuned library code */
#if 0
    dwmve0_gs(&a[irs], &lda, &nn, &m, &ja[irs], xv, &work[irs]);
    for (local_int_t i = irs; i < ire; i++)
      xv[i] = work[i] * idiag[i];
#else
    intrin_gs_colwise_regs(irs, ire, &a[irs], &idiag[irs], lda, m, &ja[irs], xv, &work[irs]); // Originally you pass const local_int as ref; does it give better performance? ask E
    for (local_int_t i = irs; i < ire; i++)
      xv[i] = work[i] * idiag[i];
#endif
  } else {
    /* C code */
    for (local_int_t j = 0; j < m; j++) {
      #pragma _NEC shortloop
      for (local_int_t i = irs; i < ire; i++)
        work[i] -= a[i + lda * j] * xv[ja[i + lda * j]];
    }
    #pragma _NEC shortloop
    for (local_int_t i = irs; i < ire; i++)
      xv[i] = work[i] * idiag[i];
  }
}

static inline void
gs_half_sweep(const local_int_t ics, const local_int_t ice, const double *a,
              const double *idiag, const local_int_t lda, const local_int_t m,
              const local_int_t *ja, double *xv, double *work)
{
  local_int_t nn = ice - ics;

  /* Assembler code */
  if(nn>255) {
    dwmve0_gs(&a[ics], &lda, &nn, &m, &ja[ics], xv, &work[ics]);
    for (local_int_t i=ics; i<ice; i++)
      xv[i] = work[i] * idiag[i];
    /* C code */
  } else {
#pragma _NEC loop_count(13)
    for (local_int_t j=0; j<m; j++)
#pragma _NEC shortloop
      for (local_int_t i=ics; i<ice; i++)
        work[i] -= a[i+lda*j] * xv[ja[i+lda*j]];
#pragma _NEC shortloop
    for (local_int_t i=ics; i<ice; i++)
      xv[i] = work[i]* idiag[i];
  }
}

//
// Triangular solve for ELL format with 0-base column indexing, colored sparse matrix.
// 
static inline void
ell_col_b0_trsv(const local_int_t ic_min, const local_int_t ic_max,
                const local_int_t *icptr, const local_int_t *iclen,
                const double *a, const double *idiag,
                const local_int_t lda, const local_int_t *ja,
                double *xv, double *work)
{
  local_int_t ic, irs, ire, m;

  if (ic_min < ic_max) {
    for (local_int_t ic = ic_min; ic < ic_max; ic++) {
      irs = icptr[ic];		// start row for color ic
      ire = icptr[ic+1];	// end row for color ic
      m   = iclen[ic];		// max width of matrix for the ic color
      ell_b0_trsv_step(irs, ire, a, idiag, lda, m, ja, xv, work);
    }
  } else {
    for (local_int_t ic = ic_max - 1; ic >= ic_min; ic--) {
      irs = icptr[ic];		// start row for color ic
      ire = icptr[ic+1];	// end row for color ic
      m   = iclen[ic];		// max width of matrix for the ic color
      ell_b0_trsv_step(irs, ire, a, idiag, lda, m, ja, xv, work);
    }
  }
    
}


int ComputeSYMGS( const SparseMatrix  & A, const Vector & r, Vector & x, int is_first);

#endif // COMPUTESYMGS_HPP
