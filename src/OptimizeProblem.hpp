
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

#ifndef OPTIMIZEPROBLEM_HPP
#define OPTIMIZEPROBLEM_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include "CGData.hpp"
#include "ELL_OPT.hpp"
#include <sstream>
#include <fstream>

#define VLEN 256
#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif

int OptimizeProblem(SparseMatrix & A, CGData & data,  Vector & b, Vector & x, Vector & xexact);

// This helper function should be implemented in a non-trivial way if OptimizeProblem is non-trivial
// It should return as type double, the total number of bytes allocated and retained after calling OptimizeProblem.
// This value will be used to report Gbytes used in ReportResults (the value returned will be divided by 1000000000.0).

double OptimizeProblemMemoryUse(const SparseMatrix & A);

void Optimize_ReplaceMatrixDiagonal(SparseMatrix &A, double *dv);
void Optimize_CheckDone(SparseMatrix &A);

static inline double ReadCachedRefTolerance(SparseMatrix &A)
{
  double refTolerance = -1.0;
  std::stringstream fname;
  Geometry *g = A.geom;
  fname << ".ref_" << g->npx << "x" << g->npy << "x" << g->npz << "_"
        << g->nx << "x" << g->ny << "x" << g->nz;
  std::ifstream f(fname.str(), std::ios::in | std::ios::binary);
  if (f.good()) {
    f.read((char *)&refTolerance, sizeof(double));
    f.close();
  }
  return refTolerance; 
}

static inline void WriteCachedRefTolerance(SparseMatrix &A, double refTolerance)
{
  std::stringstream fname;
  Geometry *g = A.geom;
  fname << ".ref_" << g->npx << "x" << g->npy << "x" << g->npz << "_"
        << g->nx << "x" << g->ny << "x" << g->nz;
  std::ofstream f(fname.str(), std::ios::out | std::ios::binary);
  if (f.good()) {
    f.write((char *)&refTolerance, sizeof(double));
    f.close();
  }
}

template<typename T>
int vcycle(int (*recfunc)(T&), T& A){
  T* curLevelMatrix = &A;
  while (curLevelMatrix) {
          int ierr = recfunc(*curLevelMatrix);
          if (ierr) return(ierr);
          curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
  }
  return(0);
}

int GenerateOPT_STRUCT     (SparseMatrix & A);
int SetDiagonalPointer     (SparseMatrix & A);
int SetF2cOperator         (SparseMatrix & A);
int SetCommTable           (SparseMatrix & A);
int PermVector             (const local_int_t * p, const Vector & x, Vector & px);

#endif  // OPTIMIZEPROBLEM_HPP
