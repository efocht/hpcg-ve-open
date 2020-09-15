
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

#ifndef COMPUTESPMV_HPP
#define COMPUTESPMV_HPP
#include "Vector.hpp"
#include "SparseMatrix.hpp"
#include "ExchangeHalo.hpp"

#define BLKSZ (80*256)
#define EXCHGCALLS (100)
#define VLEN 256
#ifndef MIN
#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

// #define INTR

extern "C"{
  void dwmve0_spmv(const double* a, const local_int_t* lda, const local_int_t* n,
                   const local_int_t* m, const local_int_t* ja, const double* x, double* y);

  void spmv_intr_regs(const double* a, const local_int_t lda, const local_int_t n,
                   const local_int_t m, const local_int_t* ja, const double* x, double* y);                
}

//
// Naive implementation of row-wise SPMV, for correctness checks
//
static void dwmve0_spmv_simple(const double* const a, const int *lda_,
                               const int *n_, const int *m_, const int* const ja,
                               const double* const vi, double* const vo)
{
  int n = *n_;
  int m = *m_;
  int lda = *lda_;

  for (int i=0; i<n; i++) {
    double sum = 0;
    for (int j=0; j<m; j++) {
      sum += a[i+lda*j] * vi[ja[i+lda*j]];
    }
    vo[i] = sum;
  }
}


//
// y = (A + D) * x
// 
static void
ell_b0_spmv_probe(const double *const a, const double *const diag, const int lda,
                  const int n, const int m, const int* const ja,
                  const double* const x, double* const y)
{
  int exchg_calls = EXCHGCALLS;
  local_int_t esteps = ((n+BLKSZ-1)/BLKSZ + exchg_calls)/(exchg_calls + 1);
  for (local_int_t i = 0, j = 0; i < n; i += BLKSZ, j++) {
    local_int_t ist = i, iend = MIN(ist + BLKSZ, n);
    local_int_t nn = iend - ist;
    // lower and upper triangular parts of the sparse matrix A
    // dwmve0_spmv(&a[ist], &lda, &nn, &m, &ja[ist], x, &y[ist]);

#ifndef VEINTR_SPMV
  dwmve0_spmv(&a[ist], &lda, &nn, &m, &ja[ist], x, &y[ist]);
#else
  spmv_intr_regs(&a[ist], lda, nn, m, &ja[ist], x, &y[ist]);
#endif

    // diagonal part of the sparse matrix A
    if (diag) {
      for(local_int_t i = ist; i < iend; i++)
        y[i] += diag[i] * x[i];
    }
#ifndef HPCG_NO_MPI
    if (exchg_calls && j % esteps == 0) {
      ExchangeHalo_probe(0);
      exchg_calls--;
    }
#endif
  }
}

//
// y = (A + D) * x + w
// 
static void
ell_b0_spmv_add_probe(const double *const a, const double *const diag, const int lda,
                      const int n, const int m, const int* const ja,
                      const double* const x, double* const y, const double* const w)
{
  int exchg_calls = EXCHGCALLS;
  local_int_t esteps = ((n+BLKSZ-1)/BLKSZ + exchg_calls)/(exchg_calls + 1);
  for (local_int_t i = 0, j = 0; i < n; i += BLKSZ, j++) {
    local_int_t ist = i, iend = MIN(ist + BLKSZ, n);
    local_int_t nn = iend - ist;
    // lower and upper triangular parts of the sparse matrix A
    // dwmve0_spmv(&a[ist], &lda, &nn, &m, &ja[ist], x, &y[ist]);

#ifndef INTR
  dwmve0_spmv(&a[ist], &lda, &nn, &m, &ja[ist], x, &y[ist]);
#else
  spmv_intr_regs(&a[ist], lda, nn, m, &ja[ist], x, &y[ist]);
#endif

    // diagonal part of the sparse matrix A
    if (diag) {
      for(local_int_t i = ist; i < iend; i++)
        y[i] += diag[i] * x[i];
    }
    if (w) {
      for (local_int_t i = ist; i < iend; i++)
        y[i] += w[i];
    }
#ifndef HPCG_NO_MPI
    if (exchg_calls && j % esteps == 0) {
      ExchangeHalo_probe(0);
      exchg_calls--;
    }
#endif
  }
}

//
// y = (A + D) * x
// dot = y * x
// 
static void
ell_b0_spmv_fusedot_probe(const double *const a, const double *const diag, const int lda,
                          const int n, const int m, const int* const ja,
                          const double* const x, double* const y, double *dot)
{
  int exchg_calls = EXCHGCALLS;
  local_int_t esteps = ((n+BLKSZ-1)/BLKSZ + exchg_calls)/(exchg_calls + 1);

  for (local_int_t i = 0, j = 0; i < n; i += BLKSZ, j++) {
    local_int_t ist = i, iend = MIN(ist + BLKSZ, n);
    local_int_t nn = iend - ist;
    // lower and upper triangular parts of the sparse matrix A

#ifndef INTR
  dwmve0_spmv(&a[ist], &lda, &nn, &m, &ja[ist], x, &y[ist]);
#else
  spmv_intr_regs(&a[ist], lda, nn, m, &ja[ist], x, &y[ist]);
#endif

    // diagonal part of the sparse matrix A
    for(local_int_t i = ist; i < iend; i++) {
      y[i] += diag[i] * x[i];
      *dot += y[i] * x[i];
    }
#ifndef HPCG_NO_MPI
    if (exchg_calls && j % esteps == 0) {
      ExchangeHalo_probe(0);
      exchg_calls--;
    }
#endif
  }
}


int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y);
int ComputeSPMV_L( const SparseMatrix & A, Vector & x, Vector & y);
int ComputeSPMV_DotProd( const SparseMatrix & A, Vector & x, Vector & y, double &alpha, double &time_allreduce);

#endif  // COMPUTESPMV_HPP
